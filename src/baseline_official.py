"""
src/baseline_official.py — Official Popularity Baseline & Category-Aware Freshness Re-Ranker.

Assignment 2: News Recommendation System
IRE CS4.406

================================================================================
Q3 IMPLEMENTATION: Official Baseline Reproduction & Principled Improvement
================================================================================

1. OFFICIAL BASELINE REPRODUCTION (ebnerd-benchmark):
   Both EB-NeRD and MIND use the unified official benchmark formulation:
     Score(a) = 0.5 * norm(pageviews) + 0.3 * norm(inviews) + 0.2 * norm(readtime/ctr)
   where norm(x) = x / (max(x) + eps) across the catalog.
   - EB-NeRD: extracted directly from article engagement metadata (total_pageviews, total_inviews, total_read_time).
   - MIND: extracted from training behavior clicks, impressions, and candidate exposure rates.

2. PRINCIPLED IMPROVEMENT (Category-Aware Freshness Weighting):
   Adds personalized signals without heavy neural infrastructure:
     - Category Affinity: P(category(a) | H_u) over user's chronological click history H_u.
     - Category-Aware Recency Decay: Recency(a, H_u) = sum_{h in H_u, cat(h)==cat(a)} exp(-lambda * delta_t).
     - Article Freshness Weighting: Freshness(a, t_imp) = exp(-gamma * delta_days).
     
   Combined Scoring Formula:
     Score_improved(u, a, t_imp) = 0.30 * Score_base(a) + 0.45 * Affinity(u, a) + 0.25 * Recency(u, a) + 0.10 * Freshness(a, t_imp)

3. ABLATION STUDY:
   Isolates the individual contributions of:
     - M0: Official Baseline (Popularity Only)
     - M1: + Category Affinity Only
     - M2: + Category Recency Decay Only
     - M3: Full Category-Aware Freshness Weighting (M1 + M2 + Freshness)


4. STATISTICAL SIGNIFICANCE:
   Paired Bootstrap 95% Confidence Intervals:
     Delta_i = Metric_i(Improved) - Metric_i(Baseline)
   Gains are statistically significant if the 95% CI strictly excludes 0 (CI_low > 0).

CLI USAGE:
  .venv/bin/python3 src/baseline_official.py --dataset ebnerd --mode ablation --sample_size 1000
  .venv/bin/python3 src/baseline_official.py --dataset mind   --mode ablation --sample_size 1000
  .venv/bin/python3 src/baseline_official.py --dataset both   --mode ablation --sample_size 1000
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import PATHS
from src.data_loader import (
    ArticleDict,
    ImpressionList,
    load_articles,
    load_behaviors,
)
from src.evaluator import (
    bootstrap_ci,
    evaluate_all_slices,
    format_results_markdown,
    mrr_score,
    ndcg_score,
)
from src.logger import get_logger, log_section

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified Official Baseline (ebnerd-benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class OfficialBaselineRanker:
    """
    Unified Official Popularity Baseline matching ebnerd-benchmark across EB-NeRD & MIND.
    Formula:
      Score = 0.5 * norm(pageviews) + 0.3 * norm(inviews) + 0.2 * norm(readtime/ctr)
    """
    def __init__(
        self,
        dataset: str,
        articles: ArticleDict,
        train_behaviors: Optional[ImpressionList] = None,
        feature_type: str = "combined",
    ):
        self.dataset = dataset.lower()
        self.articles = articles
        self.feature_type = feature_type
        self.scores: Dict[str, float] = {}

        if self.dataset == "ebnerd":
            self._fit_ebnerd()
        elif self.dataset == "mind":
            self._fit_mind(train_behaviors)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _fit_ebnerd(self) -> None:
        """Computes official normalized popularity features for EB-NeRD."""
        pv_list, iv_list, rt_list = [], [], []

        for aid, art in self.articles.items():
            pv_list.append(float(art.get("total_pageviews", 0) or 0))
            iv_list.append(float(art.get("total_inviews", 0) or 0))
            rt_list.append(float(art.get("total_read_time", 0.0) or 0.0))

        max_pv = max(pv_list) if pv_list and max(pv_list) > 0 else 1.0
        max_iv = max(iv_list) if iv_list and max(iv_list) > 0 else 1.0
        max_rt = max(rt_list) if rt_list and max(rt_list) > 0 else 1.0

        for aid, art in self.articles.items():
            pv_norm = float(art.get("total_pageviews", 0) or 0) / max_pv
            iv_norm = float(art.get("total_inviews", 0) or 0) / max_iv
            rt_norm = float(art.get("total_read_time", 0.0) or 0.0) / max_rt

            if self.feature_type == "pageviews":
                self.scores[aid] = pv_norm
            elif self.feature_type == "inviews":
                self.scores[aid] = iv_norm
            elif self.feature_type == "readtime":
                self.scores[aid] = rt_norm
            else:  # combined
                self.scores[aid] = 0.5 * pv_norm + 0.3 * iv_norm + 0.2 * rt_norm

        log.info(f"EB-NeRD official baseline [{self.feature_type}] fitted for {len(self.articles):,} articles.")

    def _fit_mind(self, train_behaviors: Optional[ImpressionList] = None) -> None:
        """Computes official normalized popularity features for MIND."""
        click_counts = defaultdict(int)
        inview_counts = defaultdict(int)

        if train_behaviors:
            for imp in train_behaviors:
                cands = imp.get("candidates", [])
                labels = imp.get("labels", [])
                for c_id, label in zip(cands, labels):
                    inview_counts[c_id] += 1
                    if label == 1:
                        click_counts[c_id] += 1
                for h_id in imp.get("history", []):
                    click_counts[h_id] += 1
                    inview_counts[h_id] += 1

        max_clicks = max(click_counts.values()) if click_counts and max(click_counts.values()) > 0 else 1.0
        max_inviews = max(inview_counts.values()) if inview_counts and max(inview_counts.values()) > 0 else 1.0

        for aid in self.articles:
            clicks = float(click_counts.get(aid, 0))
            inviews = float(inview_counts.get(aid, 0))
            ctr = clicks / (inviews + 1.0)

            clicks_norm = clicks / max_clicks
            inviews_norm = inviews / max_inviews
            ctr_norm = ctr

            if self.feature_type == "pageviews":
                self.scores[aid] = clicks_norm
            elif self.feature_type == "inviews":
                self.scores[aid] = inviews_norm
            elif self.feature_type == "readtime":
                self.scores[aid] = ctr_norm
            else:  # combined
                self.scores[aid] = 0.5 * clicks_norm + 0.3 * inviews_norm + 0.2 * ctr_norm

        log.info(f"MIND official baseline [{self.feature_type}] fitted for {len(self.articles):,} articles.")

    def score_candidates(self, candidates: List[str]) -> List[float]:
        return [self.scores.get(c, 0.0) for c in candidates]

    def rank_impression(self, history: List[str], candidates: List[str]) -> List[float]:
        return self.score_candidates(candidates)


OfficialPopularityBaseline = OfficialBaselineRanker


# ─────────────────────────────────────────────────────────────────────────────
# 2. Principled Improvement: Category-Aware Freshness Re-Ranker
# ─────────────────────────────────────────────────────────────────────────────

class CategoryAwareFreshnessRanker:
    """
    Q3 Improved Ranker:
    Enriches the official popularity baseline with:
      1. Category Affinity: P(cat(c) | H_u)
      2. Category-Aware Recency Decay: exponential decay over similar recent clicks
      3. Article Freshness Decay: exponential time-decay since article publication
    """
    def __init__(
        self,
        base_ranker: OfficialBaselineRanker,
        articles: ArticleDict,
        mode: str = "full",
        weight_base: float = 0.30,
        weight_affinity: float = 0.45,
        weight_recency: float = 0.25,
        weight_freshness: float = 0.10,
        recency_lambda: float = 0.25,
        freshness_gamma: float = 0.05,
    ):
        self.base_ranker = base_ranker
        self.articles = articles
        self.mode = mode
        self.w_base = weight_base
        self.w_aff = weight_affinity
        self.w_rec = weight_recency
        self.w_fresh = weight_freshness
        self.rec_lambda = recency_lambda
        self.fresh_gamma = freshness_gamma

    def score_impression(
        self,
        impression: dict,
    ) -> List[float]:
        """Scores candidate articles for a single user impression."""
        history = impression.get("history", [])
        candidates = impression.get("candidates", [])
        imp_time = impression.get("impression_time")

        if not candidates:
            return []

        # 1. User category click history profile
        hist_cats = [self.articles.get(h, {}).get("category") for h in history if h in self.articles]
        hist_cats = [c for c in hist_cats if c]
        tot_cats = len(hist_cats)
        cat_counts = Counter(hist_cats)

        scores = []
        for c in candidates:
            art = self.articles.get(c, {})
            cat = art.get("category")
            base_s = self.base_ranker.scores.get(c, 0.0)

            # Signal 1: Category Affinity
            aff = (cat_counts[cat] / tot_cats) if (cat and tot_cats > 0) else 0.0

            # Signal 2: Category-Aware Recency Decay (recent clicks in same category)
            rec = 0.0
            if cat and hist_cats:
                # Iterate over recent history items (up to 10 most recent)
                for j, h_cat in enumerate(reversed(hist_cats[-10:])):
                    if h_cat == cat:
                        rec += np.exp(-self.rec_lambda * j)
                rec = min(rec, 3.0)  # Bound recency score

            # Signal 3: Article Freshness Decay
            pub_t = art.get("published_time")
            fresh = 1.0
            if imp_time and pub_t and isinstance(pub_t, datetime) and isinstance(imp_time, datetime):
                days = max(0.0, (imp_time - pub_t).total_seconds() / 86400.0)
                fresh = float(np.exp(-self.fresh_gamma * min(days, 30.0)))

            # Score computation based on ablation mode
            if self.mode == "base_only":
                final_s = base_s
            elif self.mode == "affinity_only":
                final_s = self.w_base * base_s + (1.0 - self.w_base) * aff
            elif self.mode == "recency_only":
                final_s = self.w_base * base_s + (1.0 - self.w_base) * (rec / 3.0)
            else:  # "full"
                final_s = (
                    self.w_base * base_s
                    + self.w_aff * aff
                    + self.w_rec * (rec / 3.0)
                    + self.w_fresh * fresh
                )

            scores.append(float(final_s))

        return scores

    def rank_impression(self, history: List[str], candidates: List[str]) -> List[float]:
        """Evaluator interface."""
        imp = {"history": history, "candidates": candidates, "impression_time": None}
        return self.score_impression(imp)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Paired Bootstrap Statistical Significance & Ablation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    dataset: str,
    sample_size: int = 1500,
    n_bootstrap: int = 1000,
) -> Tuple[Dict[str, Dict[str, dict]], str]:
    """
    Executes the Q3 ablation study and computes paired bootstrap 95% CIs.
    """
    log_section(log, f"Q3 ABLATION STUDY & STATISTICAL SIGNIFICANCE: {dataset.upper()}")

    # 1. Load data
    articles = load_articles(dataset, split="val")
    train_behaviors = None
    if dataset == "mind":
        try:
            train_behaviors = load_behaviors(dataset, split="train", sample_size=10_000)
        except Exception as e:
            log.warning(f"Could not load MIND train behaviors ({e})")

    behaviors = load_behaviors(dataset, split="val", sample_size=sample_size)
    valid_behaviors = [
        imp for imp in behaviors
        if imp.get("labels") and len(imp["labels"]) > 1 and sum(imp["labels"]) > 0 and sum(imp["labels"]) < len(imp["labels"])
    ]
    log.info(f"Loaded {len(articles):,} articles and {len(valid_behaviors):,} evaluation impressions.")

    # 2. Instantiate Rankers
    base_ranker = OfficialBaselineRanker(dataset, articles, train_behaviors=train_behaviors)

    models: Dict[str, CategoryAwareFreshnessRanker] = {
        "M0: Official Baseline (Popularity Only)": CategoryAwareFreshnessRanker(base_ranker, articles, mode="base_only"),
        "M1: + Category Affinity Only": CategoryAwareFreshnessRanker(base_ranker, articles, mode="affinity_only"),
        "M2: + Category Recency Only": CategoryAwareFreshnessRanker(base_ranker, articles, mode="recency_only"),
        "M3: Full Category-Aware Freshness Weighting": CategoryAwareFreshnessRanker(base_ranker, articles, mode="full"),
    }

    # 3. Score all impressions per model
    model_scores: Dict[str, List[List[float]]] = {}
    for m_name, model in models.items():
        scores_list = []
        for imp in valid_behaviors:
            s = model.score_impression(imp)
            scores_list.append(s)
        model_scores[m_name] = scores_list

    # 4. Standard Sliced Evaluation
    all_results: Dict[str, Dict[str, dict]] = {}
    for m_name, scores in model_scores.items():
        res = evaluate_all_slices(valid_behaviors, scores, articles, n_bootstrap=n_bootstrap)
        all_results[m_name] = res

    # 5. Paired Bootstrap Significance (against M0 Official Baseline)
    base_scores = model_scores["M0: Official Baseline (Popularity Only)"]
    paired_significance: Dict[str, Dict[str, Tuple[float, float, float, bool]]] = {}

    log_section(log, f"PAIRED BOOTSTRAP SIGNIFICANCE TESTS (vs. M0 Baseline) on {dataset.upper()}")

    for m_name, scores in model_scores.items():
        if m_name == "M0: Official Baseline (Popularity Only)":
            continue

        diff_auc, diff_mrr, diff_ndcg5, diff_ndcg10 = [], [], [], []
        for imp, b_sc, m_sc in zip(valid_behaviors, base_scores, scores):
            labels = imp["labels"]
            auc_b = float(roc_auc_score(labels, b_sc))
            auc_m = float(roc_auc_score(labels, m_sc))
            diff_auc.append(auc_m - auc_b)

            mrr_b = mrr_score(labels, b_sc)
            mrr_m = mrr_score(labels, m_sc)
            diff_mrr.append(mrr_m - mrr_b)

            ndcg5_b = ndcg_score(labels, b_sc, k=5)
            ndcg5_m = ndcg_score(labels, m_sc, k=5)
            diff_ndcg5.append(ndcg5_m - ndcg5_b)

            ndcg10_b = ndcg_score(labels, b_sc, k=10)
            ndcg10_m = ndcg_score(labels, m_sc, k=10)
            diff_ndcg10.append(ndcg10_m - ndcg10_b)

        sig_dict = {
            "Delta_AUC": bootstrap_ci(diff_auc, n_bootstrap=n_bootstrap),
            "Delta_MRR": bootstrap_ci(diff_mrr, n_bootstrap=n_bootstrap),
            "Delta_nDCG@5": bootstrap_ci(diff_ndcg5, n_bootstrap=n_bootstrap),
            "Delta_nDCG@10": bootstrap_ci(diff_ndcg10, n_bootstrap=n_bootstrap),
        }
        paired_significance[m_name] = {
            k: (mean_v, low, high, low > 0)
            for k, (mean_v, low, high) in sig_dict.items()
        }

    # 6. Format Markdown Report
    lines = [
        f"# Q3 Ablation Study & Significance Report: {dataset.upper()}",
        f"**Impressions Evaluated**: {len(valid_behaviors):,} | **Bootstrap Iterations**: {n_bootstrap:,}\n",
        "## 1. Ablation Model Comparison (Overall Metrics with 95% CIs)",
        "| Model Variant | AUC | MRR | nDCG@5 | nDCG@10 | ILD@5 (Diversity) | Novelty@5 |",
        "|:--------------|:---:|:---:|:------:|:-------:|:-----------------:|:---------:|",
    ]

    for m_name, res in all_results.items():
        ov = res.get("Overall", {})
        auc_str = f"{ov.get('AUC', (0,0,0))[0]:.4f} ± {(ov.get('AUC', (0,0,0))[2] - ov.get('AUC', (0,0,0))[1])/2.0:.4f}"
        mrr_str = f"{ov.get('MRR', (0,0,0))[0]:.4f} ± {(ov.get('MRR', (0,0,0))[2] - ov.get('MRR', (0,0,0))[1])/2.0:.4f}"
        ndcg5_str = f"{ov.get('nDCG@5', (0,0,0))[0]:.4f} ± {(ov.get('nDCG@5', (0,0,0))[2] - ov.get('nDCG@5', (0,0,0))[1])/2.0:.4f}"
        ndcg10_str = f"{ov.get('nDCG@10', (0,0,0))[0]:.4f} ± {(ov.get('nDCG@10', (0,0,0))[2] - ov.get('nDCG@10', (0,0,0))[1])/2.0:.4f}"
        ild_str = f"{ov.get('ILD@5', (0,0,0))[0]:.4f} ± {(ov.get('ILD@5', (0,0,0))[2] - ov.get('ILD@5', (0,0,0))[1])/2.0:.4f}"
        nov_str = f"{ov.get('Novelty@5', (0,0,0))[0]:.4f} ± {(ov.get('Novelty@5', (0,0,0))[2] - ov.get('Novelty@5', (0,0,0))[1])/2.0:.4f}"

        bold = "**" if "Full" in m_name else ""
        lines.append(f"| {bold}{m_name}{bold} | {auc_str} | {mrr_str} | {ndcg5_str} | {ndcg10_str} | {ild_str} | {nov_str} |")

    lines.append("\n## 2. Statistical Significance: Paired Bootstrap 95% CIs (vs. M0 Baseline)")
    lines.append("| Improved Model Variant | Δ AUC (95% CI) | Δ MRR (95% CI) | Δ nDCG@5 (95% CI) | Δ nDCG@10 (95% CI) | Excludes 0? |")
    lines.append("|:-----------------------|:--------------:|:--------------:|:-----------------:|:------------------:|:-----------:|")

    for m_name, sig in paired_significance.items():
        d_auc = sig["Delta_AUC"]
        d_mrr = sig["Delta_MRR"]
        d_ndcg5 = sig["Delta_nDCG@5"]
        d_ndcg10 = sig["Delta_nDCG@10"]

        auc_str = f"{d_auc[0]:+.4f} [{d_auc[1]:+.4f}, {d_auc[2]:+.4f}]"
        mrr_str = f"{d_mrr[0]:+.4f} [{d_mrr[1]:+.4f}, {d_mrr[2]:+.4f}]"
        ndcg5_str = f"{d_ndcg5[0]:+.4f} [{d_ndcg5[1]:+.4f}, {d_ndcg5[2]:+.4f}]"
        ndcg10_str = f"{d_ndcg10[0]:+.4f} [{d_ndcg10[1]:+.4f}, {d_ndcg10[2]:+.4f}]"

        inert = all(
            d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0
            for d in (d_auc, d_mrr, d_ndcg5, d_ndcg10)
        )
        if inert:
            sig_flag = "⚪ **Null (feature inert)**"
        elif d_auc[3] and d_mrr[3]:
            sig_flag = "✅ **YES (p < 0.05)**"
        else:
            sig_flag = "⚠️ Partial (CI straddles 0)"

        lines.append(f"| **{m_name}** | {auc_str} | {mrr_str} | {ndcg5_str} | {ndcg10_str} | {sig_flag} |")

    report_md = "\n".join(lines)
    print("\n" + report_md + "\n")
    log.info("\n" + report_md)

    return all_results, report_md


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Official Baseline & Q3 Improvement Ablation")
    parser.add_argument("--dataset", choices=["ebnerd", "mind", "both"], default="both", help="Dataset(s)")
    parser.add_argument("--mode", choices=["eval", "ablation"], default="ablation", help="Execution mode")
    parser.add_argument("--sample_size", type=int, default=1500, help="Number of impressions")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap resamples")
    args = parser.parse_args()

    target_datasets = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]

    for ds in target_datasets:
        if args.mode == "ablation":
            run_ablation_study(dataset=ds, sample_size=args.sample_size, n_bootstrap=args.n_bootstrap)
        else:
            articles = load_articles(ds, split="val")
            train_behaviors = load_behaviors(ds, split="train", sample_size=10_000) if ds == "mind" else None
            baseline = OfficialBaselineRanker(ds, articles, train_behaviors=train_behaviors)
            val_behaviors = load_behaviors(ds, split="val", sample_size=args.sample_size)
            valid_behaviors = [imp for imp in val_behaviors if len(imp.get("candidates", [])) > 0]
            scores = [baseline.rank_impression(imp.get("history", []), imp.get("candidates", [])) for imp in valid_behaviors]
            res = evaluate_all_slices(valid_behaviors, scores, articles, n_bootstrap=args.n_bootstrap)
            md = format_results_markdown(res, dataset_name=ds, model_name="OfficialPopularityBaseline")
            print("\n" + md + "\n")


if __name__ == "__main__":
    main()
