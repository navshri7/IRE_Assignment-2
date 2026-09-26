"""
src/feature_engineering.py — 20-feature matrix builder with behavior-window boundary.

Assignment 2: News Recommendation System
IRE CS4.406

Features (20 total, exact order from src.config.FEATURE_NAMES):
 1. bm25_score            — BM25 score of candidate against user history
 2. dense_score           — Dense semantic cosine similarity
 3. hybrid_score          — RRF combination score
 4. category_match        — 1.0 if candidate category is user's top history category
 5. category_affinity     — P(candidate_category | user_history)
 6. history_len           — Length of user history, clipped at max_history_len
 7. recency_score         — Time-decayed click weight sum: Σ exp(-lambda * delta_hours)
 8. avg_read_time_hist    — Mean read time from history
 9. avg_scroll_hist       — Mean scroll percentage from history
10. click_overlap         — 1.0 if candidate article was previously clicked
11. freshness_days        — Days since article publication (clipped [0, 365])
12. freshness_log         — log1p(freshness_days)
13. popularity_log        — log1p(total_pageviews)
14. inview_rate           — total_inviews / (total_pageviews + 1)
15. readtime_rate         — total_read_time / (total_inviews + 1)
16. sentiment_score       — Article sentiment score (0.0 for MIND)
17. position_in_impression— 0-indexed position in candidates list
18. impression_size       — Total number of candidates in impression
19. is_subscriber         — 1.0 if subscriber else 0.0
20. user_cat_entropy      — Shannon entropy of category distribution in history

CLI USAGE:
  .venv/bin/python3 src/feature_engineering.py --dataset ebnerd --split val --sample_size 500
  .venv/bin/python3 src/feature_engineering.py --dataset mind --split val --sample_size 500

"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import pointbiserialr
from tqdm import tqdm

from src.config import (
    FEATURE_NAMES,
    HYPERPARAMS,
    N_BASE_FEATURES,
    N_FEATURES,
    PATHS,
    SESSION_FEATURE_NAMES,
)
from src.data_loader import ArticleDict, ImpressionList, load_articles, load_behaviors
from src.logger import get_logger
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_dense_retriever

log = get_logger(__name__)


def compute_shannon_entropy(counts: Counter) -> float:
    """Computes Shannon entropy (base e) of a frequency distribution."""
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy)


class FeatureExtractor:
    """
    Constructs the 20-feature matrix for ranking models from impressions and articles.
    Encapsulates BM25, Dense, and Hybrid retrievers for signal generation.
    """
    def __init__(
        self,
        articles: ArticleDict,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        hybrid: Optional[HybridRetriever] = None,
        top_k: Optional[int] = None,
        session_context: Optional[Dict[int, dict]] = None,
    ):
        self.articles = articles
        self.bm25 = bm25 or BM25Retriever(
            articles,
            k1=HYPERPARAMS.get("bm25_k1", 1.5),
            b=HYPERPARAMS.get("bm25_b", 0.75),
        )
        self.dense = dense or build_dense_retriever(articles)
        self.hybrid = hybrid or HybridRetriever(
            self.bm25, self.dense, rrf_c=HYPERPARAMS.get("rrf_k", 60.0)
        )

        self.max_hist_len = HYPERPARAMS.get("max_history_len", 50)
        self.decay_lambda = HYPERPARAMS.get("recency_decay_lambda", 0.1)

        # ── Stage 1 wiring ──────────────────────────────────────────────────
        # top_k=None  -> re-rank the impression's own (inview) candidate set.
        # top_k=int   -> run Stage 1 (Hybrid RRF) over the FULL catalog to build a
        #                top-K candidate pool, then re-rank that pool (true
        #                two-stage retrieve-then-rank). Offline evaluation only —
        #                see `assert_retrieval_mode_allowed`.
        self.top_k = top_k

        # Stage-1 bookkeeping. Refreshed per extract_impression_features() call;
        # accumulated across a dataset pass (reset with reset_stats()).
        self.last_stage1_stats: Dict[str, float] = {"stage1": 0.0}
        self._recall_sum: float = 0.0
        self._recall_n: int = 0

        # ── Stage 2b: within-session context (Q1.2) ─────────────────────────
        # Optional. When None (or when the impression has no session_id, as on
        # MIND), all eight session features are 0.0.
        self.session_context = session_context or {}
        self._articles = articles

    def session_features(self, impression: dict, candidates: List[str]) -> List[float]:
        """
        Within-session features (Q1.2), in SESSION_FEATURE_NAMES order.

        Returns 8 floats, all 0.0 when no session context exists for the
        impression. `seen_articles` and `prior_click_categories` are strict
        prefixes of the session, so no future or self information is used.
        """
        zero = [0.0] * len(SESSION_FEATURE_NAMES)
        if not self.session_context:
            return zero

        ctx = self.session_context.get(impression.get("impression_id"))
        if not ctx:
            return zero

        prior_imps = float(ctx.get("prior_impressions", 0))
        prior_clicks = float(ctx.get("prior_clicks", 0))
        prior_inviews = float(ctx.get("prior_inviews", 0))
        prior_ctr = (prior_clicks / prior_inviews) if prior_inviews > 0 else 0.0

        # Repetition / fatigue: how much of this slate was already shown.
        seen = ctx.get("seen_articles") or []
        seen_set = set(seen)
        overlap = (len(seen_set & set(candidates)) / len(candidates)) if candidates else 0.0

        # Category affinity of the clicks already spent in this session.
        cat_counts: Counter = ctx.get("prior_click_categories") or {}
        cat_total = sum(cat_counts.values())
        if cat_total > 0 and candidates:
            cats = [self._articles.get(a, {}).get("category", "") for a in candidates]
            hits = sum(1 for c in cats if c and cat_counts.get(c, 0) > 0)
            cat_match = hits / len(candidates)
        else:
            cat_match = 0.0

        return [
            float(ctx.get("index", 0)),
            prior_imps,
            prior_clicks,
            prior_ctr,
            float(ctx.get("prior_read_time_mean", 0.0)),
            float(ctx.get("prior_scroll_mean", 0.0)),
            float(overlap),
            float(cat_match),
        ]

    @property
    def last_stage1_recall(self) -> float:
        """Mean Stage-1 recall@K accumulated since the last reset_stats()."""
        return (self._recall_sum / self._recall_n) if self._recall_n else float("nan")

    def reset_stats(self) -> None:
        """Clears accumulated Stage-1 recall statistics."""
        self._recall_sum = 0.0
        self._recall_n = 0
        self.last_stage1_stats = {"stage1": 0.0}

    def resolved_pool(self, impression: dict) -> Tuple[List[str], Optional[np.ndarray]]:
        """Convenience: the candidate pool + labels Stage 2 actually re-ranks."""
        cands, y, _ = self.resolve_candidates(impression)
        return cands, y

    # ─────────────────────────────────────────────────────────────────────────
    # Behaviour-window boundary (single source of truth)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_behaviour_window(self, impression: dict):
        """
        Enforces the behaviour-window boundary: no history event with
        timestamp > impression_time may enter any feature.

        This is the ONLY place history is filtered. Both Stage-1 retrieval
        (the user query) and Stage-2 features consume its output, so a future
        click can never reach the query, the recency weights, or the profile.

        Returns: (history, history_times, history_read_times, history_scroll)
        """
        raw_history = impression.get("history", [])
        hist_times = impression.get("history_times", [])
        hist_read_times = impression.get("history_read_times", [])
        hist_scroll = impression.get("history_scroll", [])
        imp_time = impression.get("impression_time")

        if imp_time and hist_times and len(hist_times) == len(raw_history):
            keep = [i for i, t in enumerate(hist_times) if t is None or t <= imp_time]
            return (
                [raw_history[i] for i in keep],
                [hist_times[i] for i in keep],
                [hist_read_times[i] for i in keep] if len(hist_read_times) == len(raw_history) else [],
                [hist_scroll[i] for i in keep] if len(hist_scroll) == len(raw_history) else [],
            )

        return raw_history, hist_times, hist_read_times, hist_scroll

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: candidate generation over the full catalog
    # ─────────────────────────────────────────────────────────────────────────

    def stage1_retrieve(self, history: List[str], k: Optional[int] = None) -> List[str]:
        """
        Stage 1 candidate generation: Hybrid RRF (BM25 + Dense) top-K over the
        whole article catalog, given an already behaviour-window-filtered history.
        """
        k = int(k or self.top_k or HYPERPARAMS.get("top_k_candidates", 200))
        return [aid for aid, _ in self.hybrid.retrieve_topk(history, k=k)]

    def clicked_set(self, impression: dict) -> set:
        """Ground-truth clicked article IDs for a labelled impression."""
        cands = impression.get("candidates", [])
        labels = impression.get("labels")
        if labels is None:
            return set()
        return {c for c, l in zip(cands, labels) if l == 1}

    def resolve_candidates(
        self,
        impression: dict,
    ) -> Tuple[List[str], Optional[np.ndarray], Dict[str, float]]:
        """
        Resolves the candidate set that Stage 2 will re-rank, plus its labels.

        Returns (candidates, y, stage1_stats) where stage1_stats carries
        recall bookkeeping used by the two-stage evaluation harness.
        """
        imp_cands = list(impression.get("candidates", []))
        imp_labels = impression.get("labels")

        # ── Mode A: in-impression re-ranking (no Stage 1) ──────────────────
        if self.top_k is None:
            y = np.array(imp_labels, dtype=np.int8) if imp_labels is not None else None
            return imp_cands, y, {"stage1": 0.0, "n_candidates": len(imp_cands)}

        # ── Mode B: true two-stage — retrieve top-K, then re-rank ───────────
        history, _, _, _ = self.apply_behaviour_window(impression)
        retrieved = self.stage1_retrieve(history, k=self.top_k)

        if not retrieved:
            # Degenerate history (no query tokens): fall back to the impression
            # pool so the impression is still scorable.
            retrieved = imp_cands[: self.top_k]

        clicked = self.clicked_set(impression)
        y = (
            np.array([1 if aid in clicked else 0 for aid in retrieved], dtype=np.int8)
            if imp_labels is not None
            else None
        )

        stats = {
            "stage1": 1.0,
            "n_candidates": len(retrieved),
            "stage1_recall": (len(clicked & set(retrieved)) / len(clicked)) if clicked else float("nan"),
        }
        if not np.isnan(stats["stage1_recall"]):
            self._recall_sum += stats["stage1_recall"]
            self._recall_n += 1
        return retrieved, y, stats

    @staticmethod
    def assert_retrieval_mode_allowed(top_k: Optional[int], split: str) -> None:
        """
        Guard: Codabench submission files must be a permutation of each
        impression's *inview* candidates. Stage-1 global retrieval produces
        articles outside that set, so it is only valid for offline evaluation.
        """
        if top_k is not None and split == "test":
            raise ValueError(
                f"Stage-1 global retrieval (top_k={top_k}) cannot be used for split='test'. "
                "Codabench requires ranking exactly the impression's inview candidate set, "
                "so retrieved articles outside that set make the submission invalid. "
                "Use top_k=None for test prediction; top_k is for offline two-stage evaluation."
            )

    def extract_impression_features(
        self,
        impression: dict,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Extracts feature vectors for the candidate set of a single impression.

        The candidate set comes from `resolve_candidates`, i.e. either the
        impression's own inview candidates, or a Stage-1 top-K pool when
        `self.top_k` is set.

        Returns:
            X_imp: (num_candidates, 20) float32 array
            y_imp: (num_candidates,) int8 array or None
            cands: list of candidate article IDs
        """
        candidates, y_imp, stats = self.resolve_candidates(impression)
        self.last_stage1_stats = stats
        if not candidates:
            return np.empty((0, N_FEATURES), dtype=np.float32), None, []

        imp_time = impression.get("impression_time")
        is_sub = 1.0 if impression.get("is_subscriber") is True else 0.0
        n_cands = len(candidates)

        # ── Data Leakage Guard (behaviour-window boundary) ─────────────────
        history, filtered_times, filtered_reads, filtered_scrolls = (
            self.apply_behaviour_window(impression)
        )

        hist_set = set(history)
        h_len = min(len(history), self.max_hist_len)

        # User category profiling from history
        user_cat_counts: Counter = Counter()
        for aid in history:
            cat = self.articles.get(aid, {}).get("category", "")
            if cat:
                user_cat_counts[cat] += 1

        top_user_cat = user_cat_counts.most_common(1)[0][0] if user_cat_counts else ""
        total_hist_cats = sum(user_cat_counts.values())
        user_cat_entropy = compute_shannon_entropy(user_cat_counts)

        # Recency score computation
        recency_score = 0.0
        if imp_time and filtered_times and any(filtered_times):
            for t in filtered_times:
                if t is not None:
                    delta_hours = max(0.0, (imp_time - t).total_seconds() / 3600.0)
                    recency_score += math.exp(-self.decay_lambda * delta_hours)
        elif history:
            # Fallback exponential decay over click recency sequence
            for idx in range(len(history)):
                recency_score += math.exp(-self.decay_lambda * idx)

        # Average read time and scroll depth from history
        valid_reads = [r for r in filtered_reads if r is not None and r > 0]
        avg_read = float(np.mean(valid_reads)) if valid_reads else 0.0

        valid_scrolls = [s for s in filtered_scrolls if s is not None and s > 0]
        avg_scroll = float(np.mean(valid_scrolls)) if valid_scrolls else 0.0

        # ── Retrieval Scores (BM25, Dense, Hybrid) ──────────────────────────
        query_tokens = self.bm25.build_user_query(history)
        bm25_scores = self.bm25.score_candidates(query_tokens, candidates)

        # Numpy dense path: numerically identical to the torch path (verified
        # to ~1e-7) but avoids 4 MPS kernel launches per impression, which
        # dominated the profile on the 13.5M-row test set.
        user_emb = self.dense.compute_user_embedding_np(history)
        dense_scores = self.dense.score_candidates_np(user_emb, candidates)

        # Direct RRF fusion from existing scores
        bm25_arr = np.array(bm25_scores)
        dense_arr = np.array(dense_scores)
        bm25_order = np.argsort(-bm25_arr)
        bm25_ranks = np.empty_like(bm25_order)
        bm25_ranks[bm25_order] = np.arange(len(bm25_scores))

        dense_order = np.argsort(-dense_arr)
        dense_ranks = np.empty_like(dense_order)
        dense_ranks[dense_order] = np.arange(len(dense_scores))

        rrf_c = float(HYPERPARAMS.get("rrf_k", 60.0))
        hybrid_scores = (1.0 / (rrf_c + bm25_ranks)) + (1.0 / (rrf_c + dense_ranks))

        # ── Build Feature Matrix ────────────────────────────────────────────
        X_imp = np.zeros((n_cands, N_FEATURES), dtype=np.float32)

        # Within-session features (Q1.2) are constant across the slate, so
        # compute once and broadcast.
        sess = self.session_features(impression, candidates)

        for pos, aid in enumerate(candidates):
            art = self.articles.get(aid, {})
            cat = art.get("category", "")

            # Category match & affinity
            cat_match = 1.0 if (cat and cat == top_user_cat) else 0.0
            cat_affinity = (user_cat_counts[cat] / total_hist_cats) if (cat and total_hist_cats > 0) else 0.0

            # Freshness
            pub_time = art.get("published_time")
            if imp_time and pub_time:
                fresh_days = max(0.0, min(365.0, (imp_time - pub_time).total_seconds() / 86400.0))
            else:
                fresh_days = 0.0
            fresh_log = math.log1p(fresh_days)

            # Popularity & interaction metrics
            pageviews = float(art.get("total_pageviews", 0) or 0)
            inviews = float(art.get("total_inviews", 0) or 0)
            read_time = float(art.get("total_read_time", 0.0) or 0.0)
            sentiment = float(art.get("sentiment_score", 0.0) or 0.0)

            pop_log = math.log1p(pageviews)
            inview_rate = inviews / (pageviews + 1.0)
            readtime_rate = read_time / (inviews + 1.0)

            # Populate row
            X_imp[pos, 0]  = float(bm25_scores[pos])
            X_imp[pos, 1]  = float(dense_scores[pos])
            X_imp[pos, 2]  = float(hybrid_scores[pos]) if pos < len(hybrid_scores) else 0.0
            X_imp[pos, 3]  = cat_match
            X_imp[pos, 4]  = cat_affinity
            X_imp[pos, 5]  = float(h_len)
            X_imp[pos, 6]  = float(recency_score)
            X_imp[pos, 7]  = avg_read
            X_imp[pos, 8]  = avg_scroll
            X_imp[pos, 9]  = 1.0 if aid in hist_set else 0.0
            X_imp[pos, 10] = fresh_days
            X_imp[pos, 11] = fresh_log
            X_imp[pos, 12] = pop_log
            X_imp[pos, 13] = inview_rate
            X_imp[pos, 14] = readtime_rate
            X_imp[pos, 15] = sentiment
            X_imp[pos, 16] = float(pos)
            X_imp[pos, 17] = float(n_cands)
            X_imp[pos, 18] = is_sub
            X_imp[pos, 19] = user_cat_entropy
            # 20-27: within-session context
            if sess:
                X_imp[pos, N_BASE_FEATURES:] = sess

        return X_imp, y_imp, candidates


def extract_features_dataset(
    extractor: FeatureExtractor,
    behaviors: ImpressionList,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[int], List[str]]:
    """
    Extracts features across a full list of impressions.

    Returns:
        X             : (N_pairs, 20) float32 matrix
        y             : (N_pairs,) int8 labels or None
        groups        : (M_impressions,) int group sizes (for LightGBM LambdaRank)
        impression_ids: list of impression IDs per candidate row
        candidate_ids : list of article IDs per candidate row
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    groups: List[int] = []
    imp_ids: List[int] = []
    cand_ids: List[str] = []

    extractor.reset_stats()
    iterator = tqdm(behaviors, desc="Extracting Features", disable=not show_progress)
    has_labels = True

    for imp in iterator:
        X_imp, y_imp, cands = extractor.extract_impression_features(imp)
        if len(cands) == 0:
            continue

        X_list.append(X_imp)
        groups.append(len(cands))
        imp_id = imp["impression_id"]
        imp_ids.extend([imp_id] * len(cands))
        cand_ids.extend(cands)

        if y_imp is not None:
            y_list.append(y_imp)
        else:
            has_labels = False

    if not X_list:
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            None,
            np.empty((0,), dtype=np.int32),
            [],
            [],
        )

    X = np.vstack(X_list).astype(np.float32)
    # Replace any potential NaNs or Infs safely
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.concatenate(y_list).astype(np.int8) if has_labels and y_list else None
    groups_arr = np.array(groups, dtype=np.int32)

    return X, y, groups_arr, imp_ids, cand_ids


def print_feature_sanity(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """
    Performs checks on NaN/Inf counts, shapes, and point-biserial correlations.
    """
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    log.info(f"[FEATURE SANITY] NaN count: {nan_count} | Inf count: {inf_count}")
    log.info(f"[FEATURE SANITY] Matrix shape: {X.shape}")

    if y is not None:
        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        log.info(f"[FEATURE SANITY] Labels: {pos_count:,} positive, {neg_count:,} negative (pos_rate: {pos_count/len(y):.4f})")

        log.info("[FEATURE CORR] Point-Biserial Correlations with Target Label:")
        for idx, name in enumerate(FEATURE_NAMES):
            col = X[:, idx]
            std = float(np.std(col))
            if std > 1e-8:
                try:
                    rpb, pval = pointbiserialr(y, col)
                    log.info(f"  {name:<24}: r_pb={rpb:+.4f} (p={pval:.2e})")
                except Exception:
                    log.info(f"  {name:<24}: [constant / numerical error]")
            else:
                log.info(f"  {name:<24}: [constant feature std=0.0]")


def build_and_save_features(
    dataset: str,
    split: str = "train",
    sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Convenience function: loads data, extracts features, checks sanity, and saves to results/features/.
    """
    out_dir = output_dir or PATHS["features"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"  Phase 3: Feature Engineering — {dataset.upper()} [{split}]")
    log.info("=" * 60)

    articles = load_articles(dataset, split=split)
    behaviors = load_behaviors(dataset, split=split, sample_size=sample_size)

    extractor = FeatureExtractor(articles)
    X, y, groups, imp_ids, cand_ids = extract_features_dataset(extractor, behaviors)

    print_feature_sanity(X, y)

    # Save to disk
    prefix = f"{dataset}_{split}"
    if sample_size:
        prefix += f"_sample{sample_size}"

    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_groups.npy", groups)
    if y is not None:
        np.save(out_dir / f"{prefix}_y.npy", y)

    log.info(f"Features saved to {out_dir / prefix}_*.npy")
    return X, y, groups


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Feature Engineering")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="ebnerd", help="Dataset to extract features from")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split to process")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of impressions to process (for testing)")
    args = parser.parse_args()

    build_and_save_features(
        dataset=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
