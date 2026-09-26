"""
run_all_experiments.py — Master Experiment Runner & End-to-End Benchmark.

Assignment 2: News Recommendation System
IRE CS4.406

Runs and compares ALL models across EB-NeRD and MIND datasets:
  1. BM25 Baseline
  2. Dense SVD Baseline
  3. Hybrid RRF Fusion
  4. Official Popularity Baseline
  5. Official NRMS Neural Baseline (pure PyTorch reproduction)
  6. LightGBM LambdaRank (20 engineered features)
  7. PyTorch MLP Re-Ranker (20 engineered features)

Computes:
  - Overall Ranking Metrics (AUC, MRR, nDCG@5, nDCG@10) with 95% Bootstrap CIs
  - Sliced Evaluation (Cold vs Warm users, Head vs Tail articles)
  - Beyond-Accuracy Metrics (ILD@5, Novelty@5, Catalog Coverage@5)
  - Unified Markdown summary table saved to results/summary_<timestamp>.md
  - Optional Codabench test set submission generation (--mode test or both)

CLI USAGE:
  # Quick smoke/debug run
  .venv/bin/python3 run_all_experiments.py --dataset ebnerd --sample_size 200 --mode val

  # Full validation benchmark
  .venv/bin/python3 run_all_experiments.py --dataset both --sample_size 1000 --mode val

  # Full submission generation
  .venv/bin/python3 run_all_experiments.py --dataset both --mode test

  # Complete benchmark + submission
  .venv/bin/python3 run_all_experiments.py --dataset both --sample_size 1000 --mode both
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from src.baseline_official import (
    CategoryAwareFreshnessRanker,
    OfficialBaselineRanker,
    OfficialPopularityBaseline,
    run_ablation_study,
)
from src.config import DEVICE, HYPERPARAMS, PATHS, TRAIN_SAMPLE_SIZE
from src.data_loader import (
    ArticleDict,
    ImpressionList,
    load_articles,
    load_behaviors,
)
from src.evaluator import evaluate_all_slices, format_results_markdown
from src.feature_engineering import FeatureExtractor, extract_features_dataset
from src.logger import get_logger, log_section
from src.nrms_baseline import NRMSTrainer
from src.predictor import generate_submission_file
from src.reranker import LightGBMReranker, MLPReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_dense_retriever
from src.run_dir import index_run_dir, resolve_run_dir, write_manifest

log = get_logger("master_runner")


def _manifest_params(args) -> dict:
    """The CLI parameters worth recording in a run manifest."""
    keys = ("dataset", "mode", "models", "sample_size", "train_sample_size",
            "n_bootstrap", "top_k", "run_dir")
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def load_index_articles(dataset: str, split: str):
    """
    Loads the article catalogue used to BUILD THE RETRIEVAL INDEX.

    Every feature-extraction site must go through this so that bm25_score,
    dense_score and hybrid_score mean the same thing during training,
    evaluation and serving. Calling load_articles() directly with a bare split
    re-introduces the train/serve skew documented in data_loader.load_articles:
    MIND's per-split news.tsv files share only 28,460 of ~50K IDs, which shifts
    bm25_score (mean 6.98 -> 8.53) and dense_score (0.115 -> 0.140) between
    splits.
    """
    from src.config import SHARED_INDEX_CATALOG
    from src.data_loader import index_catalog_splits

    merge = index_catalog_splits(dataset, split) if SHARED_INDEX_CATALOG else None
    return load_articles(dataset, split=split, merge_splits=merge)


def build_session_context_for_split(dataset: str, split: str) -> dict:
    """
    Builds within-session context (Q1.2) for a split, or {} when unavailable.

    Only EB-NeRD exposes session_id, so MIND always returns {} and its eight
    session columns stay 0.0 (reported by the constant-column diagnostic).
    """
    import polars as pl

    from src.data_loader import build_session_context

    beh_path = PATHS.get(f"{dataset}_{split}_behaviors")
    if not beh_path or not Path(beh_path).exists() or Path(beh_path).suffix != ".parquet":
        log.info(f"[{dataset}.{split}] No parquet behaviors for session context; skipping.")
        return {}

    try:
        articles = load_index_articles(dataset, split)
        df = pl.read_parquet(beh_path)
        return build_session_context(df, articles=articles)
    except Exception as e:
        log.warning(f"[{dataset}.{split}] Session context unavailable: {e}")
        return {}


def load_or_retrain_reranker(
    dataset: str,
    kind: str,
    train_sample: int,
    session_ctx: dict,
):
    """
    Loads a checkpoint, or trains a fresh one when it is missing OR stale.

    Staleness has three independent causes and ALL must be caught:
      1. Feature COUNT differs from FEATURE_NAMES (e.g. the Q1.2 session
         features took the matrix from 20 to 28 columns).
      2. Feature SEMANTICS differ -- same count, different meaning. Swapping the
         dense backend from tfidf_svd to minilm keeps 28 columns but changes
         `dense_score` and `hybrid_score` entirely.
      3. Feature DISTRIBUTION differs because the INDEX CATALOGUE or the
         TRAINING SIZE changed. This one bit in practice: after unifying the
         index and decoupling the training size, a checkpoint trained on 300
         impressions against the old split catalogue still matched on (1) and
         (2), so it loaded silently and the re-ranker scored below its own
         strongest feature.

    The index articles are therefore loaded BEFORE the load decision, so the
    fingerprint can include the catalogue identity.
    """
    from src.reranker import LGBM_N_JOBS
    from src.retriever import feature_fingerprint

    models_dir = PATHS["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    arts = load_index_articles(dataset, "train")
    fp = feature_fingerprint(
        article_ids=list(arts.keys()),
        train_config={"impressions": train_sample, "kind": kind},
    )
    log.info(f"[{dataset}] Feature fingerprint: {fp}")

    if kind == "lgbm":
        path = models_dir / f"lgbm_{dataset}.txt"
        model = LightGBMReranker(n_jobs=LGBM_N_JOBS)
        if path.exists():
            try:
                return model.load(path, fingerprint=fp), False
            except ValueError as e:
                log.warning(f"[{dataset}] Discarding stale checkpoint: {e}")
        log.info(f"[{dataset}] Training LightGBM on [train] (sample={train_sample}) ...")
        behs = load_behaviors(dataset, split="train", sample_size=train_sample)
        X, y, g, _, _ = extract_features_dataset(
            FeatureExtractor(arts, session_context=session_ctx), behs, show_progress=False
        )
        model.fit(X, y, g)
        model.save(path, fingerprint=fp)
        return model, True

    if kind == "mlp":
        path = models_dir / f"mlp_{dataset}.pth"
        model = MLPReranker(epochs=3)
        if path.exists():
            try:
                return model.load(path, fingerprint=fp), False
            except RuntimeError as e:
                log.warning(f"[{dataset}] Discarding stale MLP checkpoint: {e}")
        log.info(f"[{dataset}] Training PyTorch MLP on [train] (sample={train_sample}) ...")
        behs = load_behaviors(dataset, split="train", sample_size=train_sample)
        X, y, _, _, _ = extract_features_dataset(
            FeatureExtractor(arts, session_context=session_ctx), behs, show_progress=False
        )
        model.fit(X, y)
        model.save(path, fingerprint=fp)
        return model, True

    raise ValueError(f"Unknown reranker kind: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Model registry: which experiments exist, and what each one needs built
# ─────────────────────────────────────────────────────────────────────────────
#
# "requires" lists models whose ARTEFACTS must be CONSTRUCTED for this one to be
# scored -- it does not mean they are also reported. Selecting `lgbm` therefore
# builds BM25 + Dense + Hybrid (the FeatureExtractor needs all three) but emits
# a single-model table, and skips NRMS training entirely.
#
# This is the difference between a selection flag and a real speedup: BM25
# indexing, the MiniLM encode and NRMS training are the expensive parts, and all
# three are now avoidable.
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "bm25":       {"name": "BM25",                                "requires": []},
    "dense":      {"name": "Dense SVD",                           "requires": []},
    "hybrid":     {"name": "Hybrid RRF",                          "requires": ["bm25", "dense"]},
    "popularity": {"name": "Official Popularity Baseline",         "requires": []},
    "nrms":       {"name": "NRMS (Official Baseline)",             "requires": []},
    "lgbm":       {"name": "LightGBM LambdaRank",                 "requires": ["bm25", "dense", "hybrid"]},
    "mlp":        {"name": "PyTorch MLP Re-Ranker",               "requires": ["bm25", "dense", "hybrid"]},
    "q3":         {"name": "Category-Aware Freshness (Q3 Improved)",
                   "requires": ["popularity"]},
}

# Order the benchmark reports models in, independent of the order requested.
MODEL_ORDER: List[str] = list(MODEL_REGISTRY.keys())


def resolve_models(selected: Optional[Sequence[str]]) -> Tuple[Set[str], Set[str]]:
    """
    Expand a --models selection into (models_to_score, models_to_build).

    Dependencies are pulled in transitively, so `--models lgbm` needs bm25 and
    dense BUILT to construct the FeatureExtractor without SCORING them.

    Raises ValueError on an unknown slug rather than silently running everything,
    which is the failure mode that makes a selection flag untrustworthy.
    """
    if not selected:
        return set(MODEL_ORDER), set(MODEL_ORDER)

    unknown = [m for m in selected if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model(s) {unknown}. Choose from: {', '.join(MODEL_ORDER)}"
        )

    to_score = {m for m in selected}
    to_build: Set[str] = set()

    # Iterate to a fixed point; dependency depth is 1 today but need not be.
    pending = list(to_score)
    while pending:
        m = pending.pop()
        if m in to_build:
            continue
        to_build.add(m)
        pending.extend(MODEL_REGISTRY[m]["requires"])

    return to_score, to_build


def run_single_dataset_benchmark(
    dataset: str,
    sample_size: int = 1000,
    n_bootstrap: int = 500,
    use_session_features: bool = True,
    train_sample_size: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, dict]], str]:
    """
    Executes multi-model evaluation benchmark for a given dataset.

    models: optional subset of MODEL_REGISTRY keys to score. Dependencies are
    built automatically. Defaults to every model.
    """
    train_sample_size = int(train_sample_size or TRAIN_SAMPLE_SIZE)
    """
    Executes full multi-model evaluation benchmark for a given dataset.
    """
    to_score, to_build = resolve_models(models)
    log_section(log, f"BENCHMARK RUN: {dataset.upper()} (Validation Sample: {sample_size})")
    log.info(
        f"Models to SCORE: {[MODEL_REGISTRY[m]['name'] for m in MODEL_ORDER if m in to_score]}"
    )
    built_only = [m for m in MODEL_ORDER if m in to_build and m not in to_score]
    if built_only:
        log.info(f"Built as dependencies only (not reported): {built_only}")

    # 1. Load Data
    log.info(f"Loading {dataset} validation data ...")
    articles_val = load_index_articles(dataset, "val")
    behaviors_val = load_behaviors(dataset, split="val", sample_size=sample_size)

    # 1b. Within-session context (Q1.2)
    session_ctx = build_session_context_for_split(dataset, "val") if use_session_features else {}
    if session_ctx:
        log.info(f"[{dataset.upper()}] Session context active for {len(session_ctx):,} impressions.")
    else:
        log.info(f"[{dataset.upper()}] No session context; 8 session features will be 0.0.")
    valid_behaviors = [
        imp for imp in behaviors_val
        if imp.get("labels") and len(imp["labels"]) > 0 and sum(imp["labels"]) > 0
    ]
    log.info(f"Loaded {len(articles_val):,} articles and {len(valid_behaviors):,} evaluation impressions.")

    # 2. Initialize Base Retrievers
    #    Only constructed when something needs them; each is independently
    #    expensive (BM25 index, MiniLM encode, RRF wrapper).
    need_bm25 = bool(to_build & {"bm25", "hybrid", "lgbm", "mlp"})
    need_dense = bool(to_build & {"dense", "hybrid", "lgbm", "mlp"})
    need_hybrid = bool(to_build & {"hybrid", "lgbm", "mlp"})

    bm25 = dense = hybrid = extractor_val = None
    if need_bm25 or need_dense or need_hybrid:
        log.info(f"Building retrieval indexes for {dataset} ...")
    if need_bm25:
        bm25 = BM25Retriever(articles_val)
    if need_dense:
        dense = build_dense_retriever(articles_val)
    if need_hybrid:
        hybrid = HybridRetriever(bm25, dense)
    if to_build & {"lgbm", "mlp"}:
        extractor_val = FeatureExtractor(
            articles_val, bm25=bm25, dense=dense, hybrid=hybrid, session_context=session_ctx
        )

    # 3. Model Scoring Pipelines
    all_results: Dict[str, Dict[str, dict]] = {}
    model_scores: Dict[str, List[List[float]]] = {}
    reported = [m for m in MODEL_ORDER if m in to_score]
    step = {m: i + 1 for i, m in enumerate(reported)}
    n_steps = len(reported)

    def _step(slug: str) -> str:
        return f"[{dataset.upper()}] Scoring Model {step[slug]}/{n_steps}: {MODEL_REGISTRY[slug]['name']} ..."

    # ── (A) BM25 Retriever Scoring ───────────────────────────────────────────
    if "bm25" in to_score:
        log.info(_step("bm25"))
        bm25_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring BM25 ({dataset})"):
            scores = bm25.rank_impression(imp.get("history", []), imp.get("candidates", []))
            bm25_scores.append(scores)
        model_scores["BM25"] = bm25_scores

    # ── (B) Dense SVD Retriever Scoring ──────────────────────────────────────
    if "dense" in to_score:
        log.info(_step("dense"))
        dense_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring Dense SVD ({dataset})"):
            scores = dense.rank_impression(imp.get("history", []), imp.get("candidates", []))
            dense_scores.append(scores)
        model_scores["Dense SVD"] = dense_scores

    # ── (C) Hybrid RRF Scoring ───────────────────────────────────────────────
    if "hybrid" in to_score:
        log.info(_step("hybrid"))
        hybrid_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring Hybrid RRF ({dataset})"):
            scores = hybrid.rank_impression(imp.get("history", []), imp.get("candidates", []))
            hybrid_scores.append(scores)
        model_scores["Hybrid RRF"] = hybrid_scores

    # ── (D) Official Popularity Baseline ─────────────────────────────────────
    pop_ranker = None
    if to_build & {"popularity", "q3"}:
        train_behaviors_pop = None
        if dataset == "mind":
            try:
                train_behaviors_pop = load_behaviors(dataset, split="train", sample_size=10_000)
            except Exception:
                pass
        pop_ranker = OfficialBaselineRanker(dataset, articles_val, train_behaviors_pop)
    if "popularity" in to_score:
        log.info(_step("popularity"))
        pop_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring Popularity ({dataset})"):
            scores = pop_ranker.rank_impression(imp.get("history", []), imp.get("candidates", []))
            pop_scores.append(scores)
        model_scores["Official Popularity Baseline"] = pop_scores

    # ── (E) Official NRMS Neural Baseline (PyTorch) ─────────────────────────
    if "nrms" in to_score:
        log.info(_step("nrms"))
        nrms_path = PATHS["models"] / f"nrms_{dataset}.pth"
        nrms_trainer = NRMSTrainer(dataset=dataset, epochs=2, device=DEVICE)

        if nrms_path.exists():
            log.info(f"Loading pre-trained NRMS model from {nrms_path} ...")
            nrms_trainer.load(nrms_path)
        else:
            log.info(f"Training lightweight NRMS model on {dataset} [train] ...")
            articles_train = load_index_articles(dataset, "train")
            behaviors_train = load_behaviors(dataset, split="train", sample_size=max(200, sample_size // 2))
            nrms_trainer.train(articles_train, behaviors_train)
            nrms_trainer.save(nrms_path)

        nrms_scores, _, _ = nrms_trainer.score_behaviors(valid_behaviors)
        model_scores["NRMS (Official Baseline)"] = nrms_scores

    # ── (F) LightGBM LambdaRank ──────────────────────────────────────────────
    session_ctx_tr = {}
    if to_build & {"lgbm", "mlp"}:
        session_ctx_tr = build_session_context_for_split(dataset, "train") if use_session_features else {}

    if "lgbm" in to_score:
        log.info(_step("lgbm"))
        lgbm, _ = load_or_retrain_reranker(
            dataset, "lgbm", train_sample_size, session_ctx_tr
        )

        lgbm_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring LightGBM ({dataset})"):
            X_imp, _, _ = extractor_val.extract_impression_features(imp)
            if len(X_imp) > 0:
                scores = lgbm.predict_scores(X_imp).tolist()
            else:
                scores = [0.0] * len(imp.get("candidates", []))
            lgbm_scores.append(scores)
        model_scores["LightGBM LambdaRank"] = lgbm_scores

    # ── (G) PyTorch MLP Re-Ranker ───────────────────────────────────────────
    if "mlp" in to_score:
        log.info(_step("mlp"))
        mlp, _ = load_or_retrain_reranker(
            dataset, "mlp", train_sample_size, session_ctx_tr
        )

        mlp_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring PyTorch MLP ({dataset})"):
            X_imp, _, _ = extractor_val.extract_impression_features(imp)
            if len(X_imp) > 0:
                scores = mlp.predict_scores(X_imp).tolist()
            else:
                scores = [0.0] * len(imp.get("candidates", []))
            mlp_scores.append(scores)
        model_scores["PyTorch MLP"] = mlp_scores

    # ── (H) Category-Aware Freshness Re-Ranker (Q3 Principled Improvement) ─
    if "q3" in to_score:
        log.info(_step("q3"))
        q3_ranker = CategoryAwareFreshnessRanker(base_ranker=pop_ranker, articles=articles_val, mode="full")
        q3_scores = []
        for imp in tqdm(valid_behaviors, desc=f"Scoring Q3 Improved ({dataset})"):
            scores = q3_ranker.score_impression(imp)
            q3_scores.append(scores)
        model_scores["Category-Aware Freshness (Q3 Improved)"] = q3_scores

    # 4. Sliced Evaluation & Bootstrap Metric Computation
    log_section(log, f"COMPUTING 95% BOOTSTRAP CONFIDENCE INTERVALS: {dataset.upper()}")
    dataset_reports = []

    for model_name, scores in model_scores.items():
        log.info(f"Evaluating slices and bootstrap CIs for {model_name} ...")
        results = evaluate_all_slices(
            impressions_data=valid_behaviors,
            all_scores=scores,
            articles=articles_val,
            n_bootstrap=n_bootstrap,
        )
        all_results[model_name] = results
        md_table = format_results_markdown(
            results=results,
            dataset_name=dataset,
            model_name=model_name,
        )
        dataset_reports.append(md_table)
        print("\n" + md_table + "\n")

    # Combine all results into single dataset markdown
    combined_report = "\n\n".join(dataset_reports)
    return all_results, combined_report


def run_two_stage_benchmark(
    dataset: str,
    top_k: int = 200,
    sample_size: int = 200,
    n_bootstrap: int = 500,
    train_sample_size: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, dict]], str]:
    """
    Genuine two-stage retrieve-then-rank evaluation (Q2.1 / Q2.4).

    Stage 1 builds a top-K pool over the FULL catalog with Hybrid RRF, then
    Stage 2 re-ranks that pool. Reports Stage-1 Recall@K plus metrics before
    (Stage-1 RRF order) and after (LightGBM re-ranking) the re-ranker.

    Impressions whose retrieved pool contains no clicked article are RETAINED
    (include_zero_positive=True) so Stage-1 failures are not silently dropped.
    """
    log_section(log, f"TWO-STAGE RETRIEVE-THEN-RANK BENCHMARK: {dataset.upper()} (K={top_k})")

    articles = load_index_articles(dataset, "val")
    behaviors = load_behaviors(dataset, split="val", sample_size=sample_size)
    valid = [
        imp for imp in behaviors
        if imp.get("labels") and 0 < sum(imp["labels"]) < len(imp["labels"])
    ]
    log.info(f"Loaded {len(articles):,} articles and {len(valid):,} evaluation impressions.")

    extractor = FeatureExtractor(articles, top_k=top_k)
    lgbm, _ = load_or_retrain_reranker(
        dataset, "lgbm", int(train_sample_size or TRAIN_SAMPLE_SIZE), {}
    )

    stage1_scores, stage2_scores = [], []
    pools, labs = [], []
    for imp in tqdm(valid, desc=f"Two-stage scoring ({dataset})"):
        cands, y = extractor.resolved_pool(imp)
        if not cands:
            continue
        X_imp, _, _ = extractor.extract_impression_features(imp)
        pools.append(cands)
        labs.append(y.tolist() if y is not None else None)
        # "Before": Stage-1 Hybrid RRF fusion score (feature index 2)
        stage1_scores.append(X_imp[:, 2].tolist())
        # "After": LightGBM re-ranker
        stage2_scores.append(lgbm.predict_scores(X_imp).tolist())

    recall = extractor.last_stage1_recall
    log.info(f"Stage-1 Recall@{top_k} = {recall:.4f}")

    all_results, reports = {}, []
    for name, scores in [
        (f"Stage 1 only (Hybrid RRF top-{top_k})", stage1_scores),
        (f"Stage 1 + Stage 2 (LightGBM LambdaRank)", stage2_scores),
    ]:
        res = evaluate_all_slices(
            impressions_data=valid,
            all_scores=scores,
            articles=articles,
            n_bootstrap=n_bootstrap,
            include_zero_positive=True,
            candidate_lists=pools,
            label_lists=labs,
            stage1_recall=recall,
        )
        all_results[name] = res
        md = format_results_markdown(res, dataset, name)
        reports.append(md)
        print("\n" + md + "\n")

    return all_results, "\n\n".join(reports)


def build_unified_comparison_table(
    all_dataset_results: Dict[str, Dict[str, Dict[str, dict]]],
) -> str:
    """
    Constructs a comparative summary table across all datasets and models.
    """
    lines = [
        "# Benchmark Results Summary: News Recommendation System",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Hardware**: `{DEVICE}`\n",
        "## Overall Model Performance Comparison (95% Bootstrap CIs)",
        "| Dataset | Model | AUC | MRR | nDCG@5 | nDCG@10 | ILD@5 (Diversity) | Novelty@5 | Catalog Coverage@5 |",
        "|:--------|:------|:---:|:---:|:------:|:-------:|:-----------------:|:---------:|:-------------------:|",
    ]

    for dataset, models_dict in all_dataset_results.items():
        for model_name, slices in models_dict.items():
            overall = slices.get("Overall", {})
            cov_val = slices.get("Coverage@5", 0.0)

            def _fmt(m: str) -> str:
                val = overall.get(m, (0.0, 0.0, 0.0))
                if isinstance(val, (tuple, list)) and len(val) == 3:
                    mean_v, low, high = val
                    half = (high - low) / 2.0
                    return f"{mean_v:.4f} ± {half:.4f}"
                elif isinstance(val, (int, float)):
                    return f"{val:.4f}"
                return "0.0000 ± 0.0000"

            auc_str = _fmt("AUC")
            mrr_str = _fmt("MRR")
            ndcg5_str = _fmt("nDCG@5")
            ndcg10_str = _fmt("nDCG@10")
            ild_str = _fmt("ILD@5")
            nov_str = _fmt("Novelty@5")
            cov_str = f"{cov_val:.4f}"

            lines.append(
                f"| **{dataset.upper()}** | **{model_name}** | {auc_str} | {mrr_str} | {ndcg5_str} | {ndcg10_str} | {ild_str} | {nov_str} | {cov_str} |"
            )

    return "\n".join(lines)


def generate_all_submissions(
    datasets: List[str],
    sample_size: Optional[int] = None,
) -> None:
    """
    Generates Codabench submission zip files for test sets.
    """
    log_section(log, "GENERATING CODABENCH TEST SET SUBMISSIONS")
    for ds in datasets:
        log.info(f"Generating submission for {ds.upper()} ...")
        generate_submission_file(
            dataset=ds,
            model_type="lgbm",
            split="test",
            sample_size=sample_size,
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Master Experiment Runner")
    parser.add_argument(
        "--dataset",
        choices=["ebnerd", "mind", "both"],
        default="both",
        help="Dataset(s) to evaluate",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Number of validation impressions to evaluate per dataset",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Bootstrap iterations for confidence intervals",
    )
    parser.add_argument(
        "--mode",
        choices=["val", "test", "both", "ablation", "two_stage", "serving_ablation"],
        default="val",
        help="Execution mode: val (benchmark), test (Codabench submissions), both, "
             "ablation (Q3 study), two_stage (Q2 retrieve-then-rank), "
             "serving_ablation (Q9 serving-feature subset)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Stage-1 top-K candidate pool size for --mode two_stage (Q2.1: K ~ 100-200)",
    )
    parser.add_argument(
        "--train_sample_size",
        type=int,
        default=None,
        help=(
            "Impressions used to TRAIN the re-rankers. Decoupled from "
            "--sample_size (which sizes only the evaluation set). Default "
            f"{TRAIN_SAMPLE_SIZE}. Deriving it from --sample_size meant the "
            "default run trained on 300 impressions while evaluating on 500, so "
            "the blend was starved and scored below its own strongest feature."
        ),
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Directory for this run's reports. Default: a fresh timestamped "
            "folder results/runs/<YYYYmmdd_HHMMSS>/, so no earlier run is ever "
            "overwritten. Pass a name to add to an existing run, e.g. "
            "--run_dir 20260926_140000 to co-locate the ablation with an "
            "earlier benchmark."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        metavar="MODEL",
        help=(
            "Which models to evaluate (default: all). Choose from: "
            + ", ".join(MODEL_REGISTRY.keys())
            + ". Dependencies are built automatically but NOT reported -- e.g. "
            "--models lgbm builds BM25+Dense+Hybrid to construct the feature "
            "extractor, and skips NRMS training entirely. "
            "Example: --models dense lgbm"
        ),
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print the model registry with each model's dependencies, then exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print(f"{'slug':<12} {'report name':<42} requires (built, not reported)")
        print("-" * 96)
        for slug in MODEL_ORDER:
            spec = MODEL_REGISTRY[slug]
            req = ", ".join(spec["requires"]) or "-"
            print(f"{slug:<12} {spec['name']:<42} {req}")
        return

    # Every mode writes into its own timestamped folder by default.
    run_dir = resolve_run_dir(args.run_dir)
    log_section(log, f"RUN DIRECTORY: {run_dir}")

    target_datasets = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]
    all_dataset_results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    detailed_reports = []

    # 0. Ablation Mode
    if args.mode == "ablation":
        for ds in target_datasets:
            _, md = run_ablation_study(
                dataset=ds, sample_size=args.sample_size, n_bootstrap=args.n_bootstrap
            )
            (run_dir / f"ablation_{ds}.md").write_text(md, encoding="utf-8")
            log.info(f"Saved {ds} ablation to {run_dir / f'ablation_{ds}.md'}")
        write_manifest(run_dir, params=_manifest_params(args),
                       files=index_run_dir(run_dir))
        log_section(log, "Q3 ABLATION PIPELINE COMPLETE")
        return

    # 0a. Q9 serving-feature-subset ablation
    if args.mode == "serving_ablation":
        from src.serving_ablation import run_serving_ablation

        for ds in target_datasets:
            run_serving_ablation(
                dataset=ds,
                sample_size=args.sample_size,
                n_bootstrap=args.n_bootstrap,
                run_dir=run_dir,
            )
        write_manifest(run_dir, params=_manifest_params(args),
                       files=index_run_dir(run_dir))
        log_section(log, "Q9 SERVING-FEATURE ABLATION COMPLETE")
        return

    # 0b. Two-Stage retrieve-then-rank mode
    if args.mode == "two_stage":
        for ds in target_datasets:
            ds_results, ds_md = run_two_stage_benchmark(
                dataset=ds,
                top_k=args.top_k,
                sample_size=args.sample_size,
                n_bootstrap=args.n_bootstrap,
                train_sample_size=args.train_sample_size,
            )
            all_dataset_results[ds] = ds_results
            detailed_reports.append(ds_md)

        two_stage_doc = "\n\n---\n\n".join(detailed_reports)
        ts_path = run_dir / "two_stage.md"
        ts_path.write_text(two_stage_doc, encoding="utf-8")
        log.info(f"Saved two-stage report to {ts_path}")
        write_manifest(run_dir, params=_manifest_params(args),
                       files=index_run_dir(run_dir))
        log_section(log, "Q2 TWO-STAGE PIPELINE COMPLETE")
        return

    # 1. Validation Benchmark
    if args.mode in ["val", "both"]:
        for ds in target_datasets:
            ds_results, ds_md = run_single_dataset_benchmark(
                dataset=ds,
                sample_size=args.sample_size,
                n_bootstrap=args.n_bootstrap,
                train_sample_size=args.train_sample_size,
                models=args.models,
            )
            all_dataset_results[ds] = ds_results
            detailed_reports.append(ds_md)

        # Build master unified summary table
        summary_table = build_unified_comparison_table(all_dataset_results)
        print("\n" + "=" * 80)
        print(" MASTER SUMMARY COMPARISON TABLE")
        print("=" * 80)
        print(summary_table + "\n")

        # Save the summary into this run's own timestamped directory.
        full_doc = summary_table + "\n\n---\n\n" + "\n\n---\n\n".join(detailed_reports)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(full_doc, encoding="utf-8")
        log.info(f"Saved master benchmark summary to: {summary_path}")

    # 2. Test Set Codabench Submissions
    if args.mode in ["test", "both"]:
        generate_all_submissions(target_datasets, sample_size=args.sample_size if args.sample_size < 1000 else None)

    # 3. Manifest last, so it indexes everything this run actually produced.
    write_manifest(run_dir, params=_manifest_params(args), files=index_run_dir(run_dir))
    log.info(f"Run artefacts: {run_dir}")

    log_section(log, "PHASE 8 EXPERIMENT PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
