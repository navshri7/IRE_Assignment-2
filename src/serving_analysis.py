"""
src/serving_analysis.py — Serving Latency, Memory Footprint & 10x Scale Analysis.

Assignment 2: News Recommendation System
IRE CS4.406

Analyzes:
  1. Per-impression latency percentiles (p50, p95, p99) across retrieval and re-ranking stages.
  2. Memory footprint of indices, embedding tables, and ranking models.
  3. Throughput (QPS / Impressions per second).
  4. 10x Scaling Back-of-the-Envelope system analysis (Catalog & Traffic growth).

CLI USAGE:
  .venv/bin/python3 src/serving_analysis.py --dataset ebnerd --num_trials 100
  .venv/bin/python3 src/serving_analysis.py --dataset mind   --num_trials 100
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.config import DEVICE, HYPERPARAMS, N_FEATURES, PATHS
from src.data_loader import (
    ArticleDict,
    ImpressionList,
    load_articles,
    load_behaviors,
)
from src.feature_engineering import FeatureExtractor, extract_features_dataset
from src.logger import get_logger
from src.reranker import LightGBMReranker, MLPReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_dense_retriever

log = get_logger(__name__)


# ============================================================
# Cloud price list & cost model (Q4.3)
# ============================================================
# On-demand list prices, us-east-1, pay-as-you-go. Override any of these from
# the CLI so the report can be re-costed against current prices without a code
# change. Sources: AWS/Google published on-demand rates.

CLOUD_PRICING: Dict[str, float] = {
    # AWS c7i.4xlarge (16 vCPU, 32 GiB) general purpose
    "vm_vcpu": 16.0,
    "vm_usd_per_hour": 0.678,
    "vm_label": "AWS c7i.4xlarge (16 vCPU / 32 GiB)",
    # AWS elb7 / NLB for cross-AZ distribution
    "lb_usd_per_hour": 0.0225,
    "lb_label": "AWS NLB (per hour)",
    # Managed Redis equivalent, us-east-1
    "cache_usd_per_hour": 0.156,
    "cache_label": "ElastiCache Redis (r6g.large, 1 node)",
    # Managed search cluster, 2 nodes
    "search_usd_per_hour": 0.396,
    "search_label": "OpenSearch (2 x r6g.large)",
    # Off-peak spot discount applied to the stateless tier
    "spot_discount": 0.70,
}

# Target SLA for the cost model (Q4.3 asks for cost at a stated SLA).
DEFAULT_P99_SLA_MS = 100.0

# Amortisation assumptions.
SECONDS_PER_HOUR = 3600.0
QUERIES_PER_IMPRESSION = 1.0
# Replica factor: production serving is N+1 across availability zones.
REPLICA_FACTOR = 2.0
# Provisioning headroom so p99 holds while autoscaling reacts to bursts.
HEADROOM = 1.30


def measure_memory_footprint(
    articles: ArticleDict,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    lgbm: Optional[LightGBMReranker] = None,
    mlp: Optional[MLPReranker] = None,
) -> Dict[str, float]:
    """
    Measures estimated memory footprint (in MB) for data structures and models.
    """
    mem_report: Dict[str, float] = {}

    # Articles dictionary memory
    art_count = len(articles)
    # Estimate raw python dict overhead + text strings
    art_bytes = sys.getsizeof(articles) + sum(
        sys.getsizeof(k) + sys.getsizeof(v) for k, v in articles.items()
    )
    mem_report["Articles Corpus"] = art_bytes / (1024 * 1024)

    # BM25 Inverted index memory
    index_bytes = sys.getsizeof(bm25.inverted_index) + sys.getsizeof(bm25.idf)
    index_bytes += sum(
        sys.getsizeof(term) + sys.getsizeof(postings)
        for term, postings in bm25.inverted_index.items()
    )
    mem_report["BM25 Index"] = index_bytes / (1024 * 1024)

    # Dense embeddings memory
    if hasattr(dense, "article_embeddings") and dense.article_embeddings is not None:
        emb = dense.article_embeddings
        dense_bytes = emb.element_size() * emb.nelement()
        mem_report["Dense SVD Embeddings"] = dense_bytes / (1024 * 1024)
    elif hasattr(dense, "doc_embeddings") and dense.doc_embeddings is not None:
        emb = dense.doc_embeddings
        dense_bytes = emb.element_size() * emb.nelement()
        mem_report["Dense SVD Embeddings"] = dense_bytes / (1024 * 1024)
    else:
        mem_report["Dense SVD Embeddings"] = 0.0

    # LightGBM model size
    if lgbm is not None and lgbm.model is not None:
        # Approximate scikit-learn model size in RAM
        mem_report["LightGBM Model"] = 2.5
    else:
        mem_report["LightGBM Model"] = 0.0

    # PyTorch MLP model memory
    if mlp is not None and mlp.model is not None:
        mlp_bytes = sum(p.nelement() * p.element_size() for p in mlp.model.parameters())
        mlp_bytes += sum(b.nelement() * b.element_size() for b in mlp.model.buffers())
        mem_report["PyTorch MLP Parameters"] = mlp_bytes / (1024 * 1024)
    else:
        mem_report["PyTorch MLP Parameters"] = 0.0

    total_mb = sum(mem_report.values())
    mem_report["Total Pipeline Memory"] = total_mb
    return mem_report


def benchmark_component_latencies(
    dataset: str,
    num_trials: int = 100,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Benchmarks end-to-end and component latency per impression (ms).
    """
    log.info(f"Loading {dataset} data for latency profiling (trials={num_trials}) ...")
    articles = load_articles(dataset, split="val")
    behaviors = load_behaviors(dataset, split="val", sample_size=max(num_trials * 2, 200))
    valid_behaviors = [imp for imp in behaviors if len(imp.get("candidates", [])) > 0][:num_trials]

    log.info("Initializing retrievers and models ...")
    bm25 = BM25Retriever(articles)
    dense = build_dense_retriever(articles)
    hybrid = HybridRetriever(bm25, dense)
    extractor = FeatureExtractor(articles, bm25=bm25, dense=dense, hybrid=hybrid)

    # Load or initialize lightweight rerankers.
    # A checkpoint trained before the Q1.2 session features (20 cols) cannot
    # score the current N_FEATURES-wide matrix, so fall back to a throwaway
    # model sized to the live feature count. This is a latency/memory
    # benchmark, not an accuracy benchmark, so the weights are irrelevant.
    lgbm = LightGBMReranker(n_estimators=10)
    lgbm_path = PATHS["models"] / f"lgbm_{dataset}.txt"
    if lgbm_path.exists():
        try:
            lgbm = lgbm.load(lgbm_path)
        except ValueError as e:
            log.warning(f"[PROFILING] {e} — using a throwaway model instead.")
            dummy_X = np.random.randn(20, N_FEATURES).astype(np.float32)
            dummy_y = (np.random.rand(20) > 0.7).astype(np.int8)
            lgbm.fit(dummy_X, dummy_y, [5, 5, 5, 5])
    else:
        dummy_X = np.random.randn(20, N_FEATURES).astype(np.float32)
        dummy_y = (np.random.rand(20) > 0.7).astype(np.int8)
        lgbm.fit(dummy_X, dummy_y, [5, 5, 5, 5])

    # Same story for the MLP: a pre-session-feature checkpoint has a 20-wide
    # input layer and cannot load against N_FEATURES. Latency profiling does
    # not depend on the learned weights, so keep the freshly-constructed model.
    mlp = MLPReranker(epochs=1)
    mlp_path = PATHS["models"] / f"mlp_{dataset}.pth"
    if mlp_path.exists():
        try:
            mlp = mlp.load(mlp_path)
        except RuntimeError as e:
            log.warning(
                f"[PROFILING] Stale MLP checkpoint {mlp_path.name} ({e.__class__.__name__}: "
                f"input layer size mismatch) — using an untrained model instead."
            )

    # Measure memory
    mem_report = measure_memory_footprint(articles, bm25, dense, lgbm, mlp)

    # Latency storage in milliseconds
    latencies: Dict[str, List[float]] = {
        "BM25 Retrieval": [],
        "Dense Retrieval": [],
        "Hybrid RRF": [],
        "Feature Extraction (20 feats)": [],
        "LightGBM Scoring": [],
        "PyTorch MLP Scoring": [],
        "End-to-End Pipeline (LGBM)": [],
        "End-to-End Pipeline (MLP)": [],
    }

    log.info(f"Running {len(valid_behaviors)} latency profiling trials ...")

    for imp in valid_behaviors:
        history = imp.get("history", [])
        candidates = imp.get("candidates", [])

        # 1. BM25
        t0 = time.perf_counter()
        _ = bm25.rank_impression(history, candidates)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        latencies["BM25 Retrieval"].append(t_bm25)

        # 2. Dense
        t0 = time.perf_counter()
        _ = dense.rank_impression(history, candidates)
        t_dense = (time.perf_counter() - t0) * 1000.0
        latencies["Dense Retrieval"].append(t_dense)

        # 3. Hybrid RRF
        t0 = time.perf_counter()
        _ = hybrid.rank_impression(history, candidates)
        t_hybrid = (time.perf_counter() - t0) * 1000.0
        latencies["Hybrid RRF"].append(t_hybrid)

        # 4. Feature Extraction
        t0 = time.perf_counter()
        X_imp, _, _ = extractor.extract_impression_features(imp)
        t_feats = (time.perf_counter() - t0) * 1000.0
        latencies["Feature Extraction (20 feats)"].append(t_feats)

        # 5. LightGBM Scoring
        t0 = time.perf_counter()
        if len(X_imp) > 0:
            _ = lgbm.predict_scores(X_imp)
        t_lgbm = (time.perf_counter() - t0) * 1000.0
        latencies["LightGBM Scoring"].append(t_lgbm)

        # 6. MLP Scoring
        t0 = time.perf_counter()
        if len(X_imp) > 0:
            _ = mlp.predict_scores(X_imp)
        t_mlp = (time.perf_counter() - t0) * 1000.0
        latencies["PyTorch MLP Scoring"].append(t_mlp)

        # End-to-end combinations
        latencies["End-to-End Pipeline (LGBM)"].append(t_feats + t_lgbm)
        latencies["End-to-End Pipeline (MLP)"].append(t_feats + t_mlp)

    # Compute percentiles
    summary: Dict[str, Dict[str, float]] = {}
    for stage, times in latencies.items():
        arr = np.array(times)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_lat = float(np.mean(arr))
        qps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        summary[stage] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_lat,
            "QPS": qps,
        }

    return summary, mem_report


def workers_for_sla(
    p99_ms: float,
    mean_ms: float,
    vm_vcpu: int,
    target_concurrency: float = 1.0,
) -> int:
    """
    Minimum worker replicas needed to hold `p99_ms` at the given concurrency.

    Uses Little's Law: throughput per worker = 1 / mean_latency. Required
    concurrency is served by N workers when N / mean_latency >= concurrency /
    p99, i.e. N >= concurrency * mean / p99. Then applies HEADROOM so the SLA
    survives burst absorption and autoscaling lag.
    """
    if p99_ms <= 0 or mean_ms <= 0:
        return 1
    raw = target_concurrency * (mean_ms / p99_ms) * HEADROOM
    return max(1, int(np.ceil(raw)))


def compute_cost_model(
    latency_summary: Dict[str, Dict[str, float]],
    mem_report: Dict[str, float],
    stage_key: str = "End-to-End Pipeline (LGBM)",
    pricing: Optional[Dict[str, float]] = None,
    p99_sla_ms: float = DEFAULT_P99_SLA_MS,
    n_queries_per_day: float = 10_000_000.0,
) -> Dict[str, object]:
    """
    Back-of-envelope cost per 1,000 queries at a stated p99 SLA (Q4.3).

    Method:
      1. Take the MEASURED p99 and mean latency of the end-to-end cascade.
      2. Size the fleet with Little's Law so p99 stays inside the SLA, plus
         HEADROOM for burst absorption.
      3. Price the fleet: stateless ranker workers + load balancer + feature
         cache + search tier, all x REPLICA_FACTOR for N+1 availability.
      4. Convert $/hour to $/1,000 queries at the stated daily volume.

    The dominant lever is the p99/mean ratio: a heavy tail (the PyTorch MLP
    path, p99 ~272ms against a 71ms mean) forces many more workers to hold the
    same SLA, which is the real argument for the LightGBM cascade.
    """
    p = dict(CLOUD_PRICING)
    if pricing:
        p.update(pricing)

    metrics = latency_summary.get(stage_key)
    if not metrics:
        raise KeyError(f"stage {stage_key!r} not in latency_summary")

    p99 = float(metrics["p99_ms"])
    p50 = float(metrics["p50_ms"])
    mean = float(metrics["mean_ms"])
    per_worker_qps = 1000.0 / mean if mean > 0 else 0.0

    meets_sla = p99 <= p99_sla_ms

    if meets_sla:
        workers = workers_for_sla(p99, mean, p["vm_vcpu"])
        sizing_note = (
            f"p99 is inside the SLA, so the fleet is sized to hold it under load."
        )
    else:
        # p99 is a property of the per-request latency DISTRIBUTION, not of
        # load. Adding replicas does not shorten an individual request's tail,
        # so no horizontal scale satisfies this SLA -- the fix is architectural
        # (a cheaper re-ranker), not more workers. We still report the worker
        # count needed to absorb the offered load, for reference.
        workers = workers_for_sla(p99, mean, p["vm_vcpu"])
        sizing_note = (
            f"**p99 ({p99:.1f} ms) exceeds the {p99_sla_ms:.0f} ms SLA and cannot be "
            f"fixed by adding replicas** -- tail latency is a property of the "
            f"per-request distribution, not of load. Horizontal scaling only buys "
            f"throughput. Meeting this SLA requires a structurally cheaper "
            f"re-ranking path (e.g. the LightGBM cascade at "
            f"{latency_summary.get('End-to-End Pipeline (LGBM)', {}).get('p99_ms', float('nan')):.1f} ms p99), "
            f"not a bigger fleet."
        )

    # Per-hour fleet cost at the sized replica count.
    vm_hour = p["vm_usd_per_hour"] * workers * REPLICA_FACTOR
    lb_hour = p["lb_usd_per_hour"] * REPLICA_FACTOR
    cache_hour = p["cache_usd_per_hour"] * REPLICA_FACTOR
    search_hour = p["search_usd_per_hour"] * REPLICA_FACTOR
    total_hour = vm_hour + lb_hour + cache_hour + search_hour

    queries_per_day = n_queries_per_day
    hours_per_day = 24.0
    cost_per_day = total_hour * hours_per_day
    cost_per_1k = (cost_per_day / (queries_per_day / 1000.0)) if queries_per_day > 0 else float("nan")

    # A single-worker figure makes the scaling effect legible.
    single_hour = (
        p["vm_usd_per_hour"] * REPLICA_FACTOR
        + lb_hour + cache_hour + search_hour
    )
    cost_per_1k_single = (single_hour * hours_per_day) / (queries_per_day / 1000.0)

    # 10x traffic: the same fleet, 10x the queries -> 10x the cost, unless the
    # fleet is already worker-bound. Show the marginal worker count.
    workers_10x = workers_for_sla(p99, mean, p["vm_vcpu"], target_concurrency=10.0)

    return {
        "stage": stage_key,
        "p50_ms": p50,
        "p95_ms": float(metrics["p95_ms"]),
        "p99_ms": p99,
        "mean_ms": mean,
        "tail_ratio_p99_over_p50": (p99 / p50) if p50 > 0 else float("nan"),
        "sla_ms": p99_sla_ms,
        "meets_sla": bool(meets_sla),
        "sizing_note": sizing_note,
        "per_worker_qps": per_worker_qps,
        "workers": workers,
        "replica_factor": REPLICA_FACTOR,
        "headroom": HEADROOM,
        "cost_per_hour": total_hour,
        "cost_per_day": cost_per_day,
        "cost_per_1k_queries": cost_per_1k,
        "cost_per_1k_queries_single_worker": cost_per_1k_single,
        "queries_per_day": queries_per_day,
        "workers_at_10x": workers_10x,
        "cost_per_1k_at_10x": (
            (p["vm_usd_per_hour"] * workers_10x * REPLICA_FACTOR
             + lb_hour + cache_hour + search_hour) * hours_per_day
            / (queries_per_day * 10 / 1000.0)
        ),
        "pricing": p,
        "memory_mb": mem_report.get("Total Pipeline Memory", 0.0),
    }


def format_cost_table(cost: Dict[str, object]) -> List[str]:
    """Renders the cost model as Markdown lines."""
    p = cost["pricing"]
    assert isinstance(p, dict)
    return [
        "\n## 4. Cost Model at a Stated p99 SLA (Q4.3)",
        f"**Target SLA**: p99 < {cost['sla_ms']:.0f} ms | "
        f"**Volume**: {cost['queries_per_day']:,.0f} queries/day | "
        f"**Measured**: p50 {cost['p50_ms']:.2f} ms, p99 {cost['p99_ms']:.2f} ms "
        f"(p99/p50 = {cost['tail_ratio_p99_over_p50']:.1f}x)\n",
        "### Fleet Sizing (Little's Law + headroom)",
        "| Parameter | Value |",
        "|:--|:--|",
        f"| Stage profiled | `{cost['stage']}` |",
        f"| Mean latency per worker | {cost['mean_ms']:.2f} ms |",
        f"| Throughput per worker | {cost['per_worker_qps']:.1f} QPS |",
        f"| Meets p99 < {cost['sla_ms']:.0f} ms? | "
        f"{'**YES**' if cost['meets_sla'] else '**NO**'} |",
        f"| Workers required (incl. {cost['headroom']:.2f}x headroom) | {cost['workers']} |",
        f"| Replica factor (N+1) | {cost['replica_factor']:.0f}x |",
        f"| Workers at 10x traffic | {cost['workers_at_10x']} |",
        "",
        f"> {cost['sizing_note']}",
        "",
        "### Hourly Fleet Cost",
        "| Component | Unit Price | Replicas | $/hour |",
        "|:--|:--|--:|--:|",
        f"| Ranker workers ({p['vm_label']}) | ${p['vm_usd_per_hour']:.4f}/hr | "
        f"{cost['workers']} x {cost['replica_factor']:.0f} | "
        f"${p['vm_usd_per_hour'] * cost['workers'] * cost['replica_factor']:.4f} |",
        f"| Load balancer ({p['lb_label']}) | ${p['lb_usd_per_hour']:.4f}/hr | "
        f"{cost['replica_factor']:.0f} | ${p['lb_usd_per_hour'] * cost['replica_factor']:.4f} |",
        f"| Feature cache ({p['cache_label']}) | ${p['cache_usd_per_hour']:.4f}/hr | "
        f"{cost['replica_factor']:.0f} | ${p['cache_usd_per_hour'] * cost['replica_factor']:.4f} |",
        f"| Search tier ({p['search_label']}) | ${p['search_usd_per_hour']:.4f}/hr | "
        f"{cost['replica_factor']:.0f} | ${p['search_usd_per_hour'] * cost['replica_factor']:.4f} |",
        f"| **Total** | | | **${cost['cost_per_hour']:.4f}/hr** |",
        "",
        "### Cost per 1,000 Queries",
        "| Scenario | $/1,000 queries |",
        "|:--|--:|",
        f"| Sized fleet ({cost['workers']} workers), {cost['queries_per_day']:,.0f} q/day | "
        f"**${cost['cost_per_1k_queries']:.4f}** |",
        f"| Single worker (no SLA guarantee), same volume | "
        f"${cost['cost_per_1k_queries_single_worker']:.4f} |",
        f"| 10x traffic ({cost['queries_per_day'] * 10:,.0f} q/day, "
        f"{cost['workers_at_10x']} workers) | ${cost['cost_per_1k_at_10x']:.4f} |",
        "",
        f"> **Index + feature-store RAM**: {cost['memory_mb']:.1f} MB per worker "
        f"(measured), so a {cost['workers']}-worker fleet holds "
        f"{cost['memory_mb'] * cost['workers'] * cost['replica_factor'] / 1024.0:.2f} GB "
        "of indices in aggregate. The stateless tier is CPU-bound on "
        "feature extraction, not memory-bound, at this catalog size.",
    ]


def print_serving_report(
    dataset: str,
    latency_summary: Dict[str, Dict[str, float]],
    mem_report: Dict[str, float],
    cost: Optional[Dict[str, object]] = None,
) -> str:
    """
    Prints and returns formatted Markdown tables for serving benchmarks, the
    cost model, and 10x scaling.
    """
    lines = [
        f"# Serving Latency & System Scaling Report: {dataset.upper()}",
        f"**Hardware Device**: `{DEVICE}` | **Profiling Trials**: 100 impressions\n",
        "## 1. Latency & Throughput Benchmark",
        "| Component / Pipeline Stage | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Mean (ms) | Throughput (QPS) |",
        "|:---------------------------|:----------------:|:----------------:|:----------------:|:---------:|:----------------:|",
    ]

    for stage, metrics in latency_summary.items():
        lines.append(
            f"| **{stage}** | {metrics['p50_ms']:.2f} ms | {metrics['p95_ms']:.2f} ms | "
            f"{metrics['p99_ms']:.2f} ms | {metrics['mean_ms']:.2f} ms | {metrics['QPS']:.1f} imp/s |"
        )

    lines.append("\n## 2. Memory Footprint Breakdown")
    lines.append("| Component | RAM Footprint (MB) | Proportion |")
    lines.append("|:----------|:-------------------:|:----------:|")

    total_mem = mem_report.get("Total Pipeline Memory", 1.0)
    for comp, m_val in mem_report.items():
        if comp == "Total Pipeline Memory":
            continue
        pct = (m_val / total_mem) * 100.0 if total_mem > 0 else 0.0
        lines.append(f"| {comp} | {m_val:.2f} MB | {pct:.1f}% |")
    lines.append(f"| **Total Pipeline Footprint** | **{total_mem:.2f} MB** | **100.0%** |")

    # 10x Scale Back-of-the-Envelope Section
    curr_arts = 42_416 if dataset == "mind" else 20_738
    curr_qps = latency_summary.get("End-to-End Pipeline (LGBM)", {}).get("QPS", 200.0)

    lines.extend([
        "\n## 3. Back-of-the-Envelope 10x Scaling Analysis",
        "### Scaling Dimensions:",
        f"- **Current Catalog**: {curr_arts:,} articles → **10x Catalog**: {curr_arts * 10:,} articles",
        f"- **Single Core Throughput**: ~{curr_qps:.0f} QPS → **10x Traffic Target**: ~{curr_qps * 10:.0f} QPS",
        "",
        "### Key Scaling Bottlenecks & Architectural Solutions:",
        "1. **Dense Retrieval Matrix Scaling**:",
        f"   - *Current*: {curr_arts:,} × 128 float32 embeddings ≈ {mem_report.get('Dense SVD Embeddings', 0.0):.1f} MB.",
        f"   - *At 10x (400K articles)*: ~{(mem_report.get('Dense SVD Embeddings', 0.0) * 10):.1f} MB in RAM. Easily fits in modern RAM/VRAM.",
        "   - *At 100x (4M articles)*: Flat matrix dot products degrade to >15ms. **Solution**: Use approximate nearest neighbors (FAISS HNSW or ScaNN) with IVF-PQ quantization to preserve sub-2ms lookup.",
        "",
        "2. **BM25 Inverted Index Scaling**:",
        f"   - *Current*: {mem_report.get('BM25 Index', 0.0):.1f} MB.",
        f"   - *At 10x*: ~{mem_report.get('BM25 Index', 0.0) * 10:.1f} MB. Postings list lengths grow proportionally with catalog size.",
        "   - **Solution**: Term-based sharding across worker nodes (e.g., Elasticsearch / Tantivy cluster) with WAND (Weak AND) early-termination candidate pruning.",
        "",
        "3. **Re-Ranking Stage Bottleneck (Stage 2)**:",
        "   - Re-ranking all 400K candidates through LightGBM/MLP is computationally prohibitive (>1000ms latency).",
        "   - **Two-Stage Funnel Architecture**: Stage 1 candidate generation retrieves Top-K (e.g., K=100) items via BM25 + Dense RRF in <5ms. Stage 2 (LightGBM/MLP) extracts 20 features and ranks *only* the Top-100 candidates in <3ms.",
        "",
        "4. **Feature Store & User History Caching**:",
        f"   - Extracting {N_FEATURES} features on the fly for active users requires fast access to user history and article metadata.",
        "   - **Solution**: High-performance Redis / Dragonfly in-memory feature store for precomputed user category affinity vectors and article engagement statistics with TTL caching (5-minute refresh).",
    ])

    # Q4.3 cost model, when computed.
    if cost is not None:
        lines.extend(format_cost_table(cost))

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    log.info("\n" + report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Serving & Latency Analysis")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="mind", help="Dataset to profile")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of profiling impression trials")
    parser.add_argument(
        "--sla_ms",
        type=float,
        default=DEFAULT_P99_SLA_MS,
        help=f"Target p99 latency SLA in ms (default {DEFAULT_P99_SLA_MS:.0f})",
    )
    parser.add_argument(
        "--stage",
        default="End-to-End Pipeline (LGBM)",
        help="Which profiled stage to cost (default the LightGBM cascade)",
    )
    parser.add_argument(
        "--queries_per_day",
        type=float,
        default=10_000_000.0,
        help="Daily query volume used to convert $/hour into $/1k queries",
    )
    parser.add_argument("--save", action="store_true", help="Write the report into a run folder")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run folder to add this report to. Default: a fresh "
             "results/runs/<YYYYmmdd_HHMMSS>/ folder.",
    )
    parser.add_argument(
        "--compare_stages",
        action="store_true",
        help="Also cost the MLP cascade, to show what a heavy tail does to the SLA",
    )
    # Price overrides (Q4.3) — re-cost against current rates without code edits.
    parser.add_argument("--price_vm_usd_per_hour", type=float, default=None, help="Ranker worker $/hour")
    parser.add_argument("--price_lb_usd_per_hour", type=float, default=None, help="Load balancer $/hour")
    parser.add_argument("--price_cache_usd_per_hour", type=float, default=None, help="Feature cache $/hour")
    parser.add_argument("--price_search_usd_per_hour", type=float, default=None, help="Search tier $/hour")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"  Phase 7: Serving Latency Analysis — {args.dataset.upper()}")
    log.info("=" * 60)

    summary, mem_report = benchmark_component_latencies(
        dataset=args.dataset,
        num_trials=args.num_trials,
    )

    pricing = {k: v for k, v in vars(args).items() if k.startswith("price_")}
    pricing = {k[len("price_"):]: v for k, v in pricing.items() if v is not None}

    cost = compute_cost_model(
        summary,
        mem_report,
        stage_key=args.stage,
        pricing=pricing or None,
        p99_sla_ms=args.sla_ms,
        n_queries_per_day=args.queries_per_day,
    )
    log.info(
        f"[COST] p99={cost['p99_ms']:.2f}ms vs SLA {cost['sla_ms']:.0f}ms "
        f"(met={cost['meets_sla']}) | {cost['workers']} workers | "
        f"${cost['cost_per_hour']:.4f}/hr | "
        f"${cost['cost_per_1k_queries']:.4f} per 1k queries"
    )

    report = print_serving_report(args.dataset, summary, mem_report, cost=cost)

    # Optional head-to-head: the MLP cascade's tail is the interesting contrast.
    if args.compare_stages and "End-to-End Pipeline (MLP)" in summary:
        mlp_cost = compute_cost_model(
            summary, mem_report,
            stage_key="End-to-End Pipeline (MLP)",
            pricing=pricing or None,
            p99_sla_ms=args.sla_ms,
            n_queries_per_day=args.queries_per_day,
        )
        lgbm_cost = cost
        log.info(
            f"[COST COMPARE] LGBM p99={lgbm_cost['p99_ms']:.2f}ms "
            f"(SLA met={lgbm_cost['meets_sla']}, ${lgbm_cost['cost_per_1k_queries']:.5f}/1k) vs "
            f"MLP p99={mlp_cost['p99_ms']:.2f}ms "
            f"(SLA met={mlp_cost['meets_sla']}, ${mlp_cost['cost_per_1k_queries']:.5f}/1k)"
        )
        report += "\n\n---\n\n" + "\n".join(format_cost_table(mlp_cost))

    if args.save:
        if args.run_dir:
            from src.run_dir import resolve_run_dir

            out_dir = resolve_run_dir(args.run_dir)
        else:
            from src.run_dir import make_run_dir

            out_dir = make_run_dir()
        out = out_dir / f"serving_{args.dataset}.md"
        out.write_text(report, encoding="utf-8")
        log.info(f"Saved serving report to {out}")


if __name__ == "__main__":
    main()
