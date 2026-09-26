"""
src/evaluator.py — Evaluation metrics, user/article slicing, and Bootstrap CIs.

Assignment 2: News Recommendation System
IRE CS4.406

REUSES (from ire-assignment1/evaluate.py):
  - Ranking metrics: dcg_score, ndcg_score, mrr_score, recall_at_k
  - Beyond-accuracy: compute_intra_list_diversity, compute_novelty, compute_coverage
  - Bootstrap CI: bootstrap_ci
  - Codabench writer: write_codabench_predictions, scores_to_ranks

EXTENDS WITH:
  - Head/Tail article slicing (Head: pageviews/popularity > median, Tail: <= median)
  - evaluate_all_slices(): computes Overall, ColdStart, Warm, Head, Tail metrics with 95% CIs
  - format_results_markdown(): pretty-prints comparative results tables in markdown
  - Standalone evaluation CLI for any model (bm25, dense, hybrid, lgbm, mlp)

CLI USAGE:
  .venv/bin/python3 src/evaluator.py --dataset mind   --model bm25 --sample_size 500
  .venv/bin/python3 src/evaluator.py --dataset ebnerd --model lgbm --sample_size 500
"""

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import HYPERPARAMS, PATHS
from src.data_loader import ArticleDict, ImpressionList, load_articles, load_behaviors
from src.logger import get_logger
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_dense_retriever
from src.feature_engineering import FeatureExtractor, extract_features_dataset
from src.reranker import LightGBMReranker, MLPReranker

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ranking & Codabench Utilities (Reused from A1)
# ─────────────────────────────────────────────────────────────────────────────

def scores_to_ranks(scores: List[float]) -> List[int]:
    """Converts prediction scores into 1-indexed ranks (highest score -> rank 1)."""
    scores_arr = np.array(scores)
    order = np.argsort(-scores_arr, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def write_codabench_predictions(
    results: List[Tuple[int, List[int]]],
    output_file: Union[str, Path],
    zip_file: Optional[Union[str, Path]] = None,
) -> None:
    """Writes results in Codabench format: `impression_id [rank1,rank2,...]` and creates zip archive."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for imp_id, ranks in results:
            f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")

    if zip_file:
        zip_path = Path(zip_file)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_path, arcname=output_path.name)
        log.info(f"Written Codabench submission zip: {zip_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Official Evaluation Metrics (Reused from A1)
# ─────────────────────────────────────────────────────────────────────────────

def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)[:k]
    y_true_sorted = y_true[order]
    gains = 2**y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: List[int], y_score: List[float], k: int = 10) -> float:
    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    if np.sum(y_true_arr) == 0:
        return 0.0
    actual_dcg = dcg_score(y_true_arr, y_score_arr, k)
    ideal_dcg = dcg_score(y_true_arr, y_true_arr, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_score(y_true: List[int], y_score: List[float]) -> float:
    order = np.argsort(-np.array(y_score))
    y_true_sorted = np.array(y_true)[order]
    hits = np.where(y_true_sorted == 1)[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], ground_truth_clicked: List[str], k: int) -> float:
    """Proportion of ground-truth clicked items present in top-K retrieved."""
    if not ground_truth_clicked:
        return 0.0
    topk_set = set(retrieved_ids[:k])
    hits = sum(1 for aid in ground_truth_clicked if aid in topk_set)
    return hits / len(ground_truth_clicked)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Beyond-Accuracy Metrics (Reused from A1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_intra_list_diversity(recommended_items: List[str], category_map: Dict[str, str]) -> float:
    """Intra-list diversity: fraction of unique categories among top recommendations."""
    if not recommended_items:
        return 0.0
    categories = [category_map.get(aid, "unknown") for aid in recommended_items]
    return len(set(categories)) / len(recommended_items)


def compute_novelty(recommended_items: List[str], item_click_counts: Dict[str, int], total_clicks: int) -> float:
    """Novelty: average self-information -log2(p(item)) of recommended items."""
    if not recommended_items or total_clicks == 0:
        return 0.0
    self_info = []
    for aid in recommended_items:
        count = item_click_counts.get(aid, 1)
        prob = count / total_clicks
        self_info.append(-np.log2(max(prob, 1e-12)))
    return float(np.mean(self_info))


def compute_coverage(all_recommended_lists: List[List[str]], total_catalog_items: int) -> float:
    """Catalog Coverage: fraction of unique articles in the catalog recommended at least once."""
    if total_catalog_items == 0:
        return 0.0
    unique_recommended = set(aid for rec_list in all_recommended_lists for aid in rec_list)
    return len(unique_recommended) / total_catalog_items


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bootstrap Confidence Intervals (Reused from A1)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    metric_values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Computes mean and 95% bootstrap confidence interval (mean, lower, upper).
    Returns (mean, lower_ci, upper_ci).
    """
    if not metric_values:
        return 0.0, 0.0, 0.0
    arr = np.array(metric_values, dtype=np.float64)
    if len(arr) <= 1:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return val, val, val

    rng = np.random.default_rng(seed=42)
    # Vectorized bootstrap resampling
    sample_matrix = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    means = np.mean(sample_matrix, axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    return float(np.mean(arr)), lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sliced Multi-Metric Evaluation Harness
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_slices(
    impressions_data: ImpressionList,
    all_scores: List[List[float]],
    articles: ArticleDict,
    n_bootstrap: int = 1000,
    include_zero_positive: bool = False,
    candidate_lists: Optional[List[List[str]]] = None,
    label_lists: Optional[List[List[int]]] = None,
    stage1_recall: Optional[float] = None,
) -> Dict[str, dict]:
    """
    Evaluates ranking performance with 95% Bootstrap CIs across slices:
      - Overall
      - ColdStart Users (History <= 5)
      - Warm Users (History > 5)
      - Head Articles (Candidate majority pageviews > median)
      - Tail Articles (Candidate majority pageviews <= median)
      - Beyond-Accuracy: ILD@5, Novelty@5, Catalog Coverage@5

    include_zero_positive:
        When True, impressions whose retrieved candidate set contains no clicked
        article are RETAINED (they contribute MRR=0 / nDCG=0). This is required
        for honest two-stage evaluation: Stage 1 genuinely fails on some
        impressions, and dropping them would inflate every metric.
        Their AUC is undefined (single-class) so it is recorded as NaN and
        excluded from AUC aggregation only.
        Leave False for in-impression re-ranking, where legacy behaviour applies.

    candidate_lists / label_lists:
        Per-impression overrides for the candidate IDs and labels that were
        actually scored. Required in two-stage mode, where the scored pool is
        the Stage-1 top-K retrieval rather than the impression's inview set.
        When omitted, the impression's own `candidates` / `labels` are used.
    """
    # Category mapping for diversity
    category_map = {aid: art.get("category", "unknown") for aid, art in articles.items()}

    # Item interaction frequencies for novelty and head/tail split
    item_click_counts: Dict[str, int] = Counter()
    for imp in impressions_data:
        for aid in imp.get("history", []):
            item_click_counts[aid] += 1

    total_clicks = max(1, sum(item_click_counts.values()))

    # Which candidate pool is actually being scored?
    pools: List[List[str]] = (
        candidate_lists
        if candidate_lists is not None
        else [imp.get("candidates", []) for imp in impressions_data]
    )
    labels: List[Optional[List[int]]] = (
        label_lists
        if label_lists is not None
        else [imp.get("labels") for imp in impressions_data]
    )

    # Calculate median popularity across all candidate items in evaluation impressions
    all_eval_pops = [
        float(articles.get(aid, {}).get("total_pageviews", 0) or item_click_counts.get(aid, 0))
        for pool in pools
        for aid in pool
    ]
    median_pop = float(np.median(all_eval_pops)) if all_eval_pops else 1.0

    per_imp_metrics = []
    cold_indices, warm_indices = [], []
    head_indices, tail_indices = [], []
    all_top5_recommendations = []

    for idx, (imp, scores) in enumerate(zip(impressions_data, all_scores)):
        y_true = labels[idx]
        if not y_true or sum(y_true) == len(y_true):
            continue
        # Impressions with no clicked article in the candidate set: a Stage-1
        # retrieval failure, not a missing sample. Keep them when asked.
        if sum(y_true) == 0 and not include_zero_positive:
            continue

        cand_list = pools[idx]
        if len(cand_list) != len(scores):
            continue

        order = np.argsort(-np.array(scores))
        top5_recs = [cand_list[i] for i in order[:5] if i < len(cand_list)]
        all_top5_recommendations.append(top5_recs)

        # Standard accuracy metrics. AUC is undefined for single-class sets.
        if sum(y_true) == 0:
            auc_val = float("nan")
        else:
            try:
                auc_val = float(roc_auc_score(y_true, scores))
            except Exception:
                auc_val = float("nan")

        mrr_val = mrr_score(y_true, scores)
        ndcg5_val = ndcg_score(y_true, scores, k=5)
        ndcg10_val = ndcg_score(y_true, scores, k=10)

        # Beyond accuracy
        ild_val = compute_intra_list_diversity(top5_recs, category_map)
        novelty_val = compute_novelty(top5_recs, item_click_counts, total_clicks)

        rec_idx = len(per_imp_metrics)
        per_imp_metrics.append({
            "AUC": auc_val,
            "MRR": mrr_val,
            "nDCG@5": ndcg5_val,
            "nDCG@10": ndcg10_val,
            "ILD@5": ild_val,
            "Novelty@5": novelty_val,
        })

        # User Slice: Cold vs Warm
        hist_len = len(imp.get("history", []))
        if hist_len <= 5:
            cold_indices.append(rec_idx)
        else:
            warm_indices.append(rec_idx)

        # Article Slice: Head vs Tail (based on mean candidate popularity)
        cand_pops = [
            float(articles.get(aid, {}).get("total_pageviews", 0) or item_click_counts.get(aid, 0))
            for aid in cand_list
        ]
        mean_cand_pop = float(np.mean(cand_pops)) if cand_pops else 0.0
        if mean_cand_pop > median_pop:
            head_indices.append(rec_idx)
        else:
            tail_indices.append(rec_idx)

    def summarize_slice(indices: List[int]) -> Dict[str, Tuple[float, float, float]]:
        summary = {}
        if not indices:
            for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@5", "Novelty@5"]:
                summary[metric] = (0.0, 0.0, 0.0)
            return summary

        for metric_name in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@5", "Novelty@5"]:
            values = [per_imp_metrics[i][metric_name] for i in indices]
            # NaN entries (undefined AUC on single-class impressions) are dropped
            # rather than coerced to 0.5, which would bias the mean downward.
            finite = [v for v in values if not np.isnan(v)]
            summary[metric_name] = bootstrap_ci(finite, n_bootstrap=n_bootstrap)
        return summary

    all_indices = list(range(len(per_imp_metrics)))
    catalog_coverage = compute_coverage(all_top5_recommendations, len(articles))

    results = {
        "Overall": summarize_slice(all_indices),
        "ColdStart": summarize_slice(cold_indices),
        "Warm": summarize_slice(warm_indices),
        "Head": summarize_slice(head_indices),
        "Tail": summarize_slice(tail_indices),
        "Coverage@5": catalog_coverage,
        "TotalEvaluated": len(per_imp_metrics),
        "Stage1Recall": stage1_recall if stage1_recall is not None else float("nan"),
    }
    return results


def format_results_markdown(
    results: Dict[str, dict],
    dataset_name: str,
    model_name: str,
) -> str:
    """Formats multi-slice evaluation dictionary into a GitHub Flavored Markdown table."""
    slices = ["Overall", "ColdStart", "Warm", "Head", "Tail"]
    metrics = ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD@5", "Novelty@5"]

    lines = [
        f"### Evaluation Results: {dataset_name.upper()} | Model: `{model_name}`",
        f"*Evaluated on {results.get('TotalEvaluated', 0):,} impressions | "
        f"Catalog Coverage@5: {results.get('Coverage@5', 0.0):.4f} | "
        f"Stage-1 Recall@K: {results.get('Stage1Recall', float('nan')):.4f}*\n",
        "| Metric | Overall (95% CI) | Cold-Start | Warm | Head Items | Tail Items |",
        "|:-------|:----------------:|:----------:|:----:|:----------:|:----------:|",
    ]

    for m in metrics:
        row = [f"**{m}**"]
        for s in slices:
            slice_dict = results.get(s, {})
            mean_val, low, high = slice_dict.get(m, (0.0, 0.0, 0.0))
            half_ci = (high - low) / 2.0
            row.append(f"{mean_val:.4f} ± {half_ci:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation and slicing")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="mind", help="Dataset to evaluate on")
    parser.add_argument("--model", choices=["bm25", "dense", "hybrid", "lgbm", "mlp"], default="bm25", help="Model to evaluate")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of impressions to evaluate")
    parser.add_argument("--split", default="val", help="Split to evaluate on")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"  Phase 5: Evaluator — {args.dataset.upper()} [{args.model}]")
    log.info("=" * 60)

    articles = load_articles(args.dataset, split=args.split)
    behaviors = load_behaviors(args.dataset, split=args.split, sample_size=args.sample_size)

    # Generate predictions
    all_scores: List[List[float]] = []

    if args.model in ["bm25", "dense", "hybrid"]:
        if args.model == "bm25":
            retriever = BM25Retriever(articles)
        elif args.model == "dense":
            retriever = build_dense_retriever(articles)
        else:
            bm25 = BM25Retriever(articles)
            dense = build_dense_retriever(articles)
            retriever = HybridRetriever(bm25, dense)

        for imp in tqdm(behaviors, desc=f"Scoring {args.model}"):
            s = retriever.rank_impression(imp.get("history", []), imp.get("candidates", []))
            all_scores.append(s)

    elif args.model in ["lgbm", "mlp"]:
        extractor = FeatureExtractor(articles)
        X, y, groups, imp_ids, cand_ids = extract_features_dataset(extractor, behaviors)

        if args.model == "lgbm":
            model_path = PATHS["models"] / f"lgbm_{args.dataset}.txt"
            reranker = LightGBMReranker()
            if model_path.exists():
                reranker.load(model_path)
            else:
                log.info(f"Model file {model_path} not found — training on val sample ...")
                reranker.fit(X, y, groups)
            flat_scores = reranker.predict_scores(X)
        else:
            model_path = PATHS["models"] / f"mlp_{args.dataset}.pth"
            reranker = MLPReranker()
            if model_path.exists():
                reranker.load(model_path)
            else:
                log.info(f"Model file {model_path} not found — training on val sample ...")
                reranker.fit(X, y)
            flat_scores = reranker.predict_scores(X)

        # Unpack flat scores per impression
        idx = 0
        valid_behaviors = [imp for imp in behaviors if len(imp.get("candidates", [])) > 0]
        for imp in valid_behaviors:
            n = len(imp.get("candidates", []))
            all_scores.append(flat_scores[idx : idx + n].tolist())
            idx += n
        behaviors = valid_behaviors

    # Compute full sliced evaluation
    results = evaluate_all_slices(behaviors, all_scores, articles)
    md_table = format_results_markdown(results, args.dataset, args.model)
    print("\n" + md_table + "\n")
    log.info("\n" + md_table)


if __name__ == "__main__":
    main()
