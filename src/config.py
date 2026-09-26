"""
src/config.py — Centralized path constants, hyperparameters, and device detection.

Assignment 2: News Recommendation System
IRE CS4.406

HOW TO USE:
    from src.config import PATHS, HYPERPARAMS, DEVICE, FEATURE_NAMES
"""

import os
import sys

# Prevent OpenMP/PyTorch MPS deadlocks on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from pathlib import Path

# ============================================================
# 1. Root Paths
# ============================================================

# Repo root: parent of this file's parent (src/ → repo_root/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---- EB-NeRD Paths (verified from filesystem) ----
EBNERD_SMALL_DIR   = REPO_ROOT / "data" / "ebnerd_small"
EBNERD_TRAIN_DIR   = EBNERD_SMALL_DIR / "train"
EBNERD_VAL_DIR     = EBNERD_SMALL_DIR / "validation"
EBNERD_ARTICLES    = EBNERD_SMALL_DIR / "articles.parquet"

EBNERD_TEST_DIR      = REPO_ROOT / "data" / "ebnerd_testset" / "test"
EBNERD_TEST_ARTICLES = REPO_ROOT / "data" / "ebnerd_testset" / "articles.parquet"

# ---- MIND Paths ----
MIND_DATA_DIR        = REPO_ROOT / "MIND_data"
MIND_SMALL_TRAIN_DIR = MIND_DATA_DIR / "MINDsmall_train"   # ~65k impressions
MIND_LARGE_TRAIN_DIR = MIND_DATA_DIR / "MINDlarge_train"   # ~2.2M impressions
MIND_TRAIN_DIR       = MIND_SMALL_TRAIN_DIR                # default — override via run_experiments --mind_size large
MIND_VAL_DIR         = MIND_DATA_DIR / "MINDsmall_dev"
MIND_TEST_DIR        = MIND_DATA_DIR / "MINDlarge_test"

# ---- A1 Code References (read-only, not committed to A2 repo) ----
A1_DIR = REPO_ROOT / "ire-assignment1"

# ---- Output Dirs ----
LOGS_DIR         = REPO_ROOT / "logs"
RESULTS_DIR      = REPO_ROOT / "results"
FEATURES_DIR     = RESULTS_DIR / "features"
MODELS_DIR       = RESULTS_DIR / "models"
SUBMISSIONS_DIR  = RESULTS_DIR / "submissions"

# Auto-create output dirs (idempotent)
for _d in [LOGS_DIR, RESULTS_DIR, FEATURES_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Structured PATHS Dict (convenience for modules)
# ============================================================

PATHS = {
    # EB-NeRD
    "ebnerd_train_behaviors" : EBNERD_TRAIN_DIR / "behaviors.parquet",
    "ebnerd_train_history"   : EBNERD_TRAIN_DIR / "history.parquet",
    "ebnerd_val_behaviors"   : EBNERD_VAL_DIR   / "behaviors.parquet",
    "ebnerd_val_history"     : EBNERD_VAL_DIR   / "history.parquet",
    "ebnerd_test_behaviors"  : EBNERD_TEST_DIR  / "behaviors.parquet",
    "ebnerd_test_history"    : EBNERD_TEST_DIR  / "history.parquet",
    "ebnerd_articles"        : EBNERD_ARTICLES,
    "ebnerd_test_articles"   : EBNERD_TEST_ARTICLES,

    # MIND (train paths are for MINDsmall by default; overrideable at runtime)
    "mind_train_behaviors"       : MIND_SMALL_TRAIN_DIR / "behaviors.tsv",
    "mind_train_news"            : MIND_SMALL_TRAIN_DIR / "news.tsv",
    "mind_large_train_behaviors" : MIND_LARGE_TRAIN_DIR / "behaviors.tsv",
    "mind_large_train_news"      : MIND_LARGE_TRAIN_DIR / "news.tsv",
    "mind_val_behaviors"         : MIND_VAL_DIR         / "behaviors.tsv",
    "mind_val_news"              : MIND_VAL_DIR         / "news.tsv",
    "mind_test_behaviors"        : MIND_TEST_DIR        / "behaviors.tsv",
    "mind_test_news"             : MIND_TEST_DIR        / "news.tsv",

    # Outputs
    "logs"       : LOGS_DIR,
    "results"    : RESULTS_DIR,
    "features"   : FEATURES_DIR,
    "models"     : MODELS_DIR,
    "submissions": SUBMISSIONS_DIR,
}

# ============================================================
# 3. Device Detection: cuda → mps → cpu
# ============================================================

def _detect_device(override: str = None) -> str:
    """Auto-detect best available device. Override with explicit string."""
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE: str = _detect_device()

# ============================================================
# 4. Hyperparameters
# ============================================================

HYPERPARAMS = {
    # BM25
    "bm25_k1"          : 1.5,
    "bm25_b"           : 0.75,
    "bm25_max_history" : 10,       # last N clicked articles for query

    "dense_backend"     : "minilm",
    "dense_model_name"  : "all-MiniLM-L6-v2",
    "dense_batch_size"  : 512,     
    "dense_torch_threads": 6,
    "dense_cache"        : True,
    "dense_encode_device": "cpu",
    "dense_use_faiss"   : True,
    "tfidf_max_features": 50_000,
    "svd_n_components"  : 128,     # embedding dim after TruncatedSVD (tfidf_svd only)

    # Hybrid RRF
    "rrf_k"            : 60,       # RRF constant

    # Candidate retrieval
    "top_k_candidates" : 200,

    # Feature Engineering
    "recency_decay_lambda": 0.1,   # exp(-lambda * delta_hours)
    "max_history_len"     : 50,    # clip history length

    # LightGBM
    "lgbm_n_estimators"  : 500,
    "lgbm_num_leaves"    : 63,
    "lgbm_learning_rate" : 0.05,
    "lgbm_min_child_samples": 10,
    "lgbm_early_stopping": 50,

    # MLP
    "mlp_hidden_dims"   : [256, 128, 64],
    "mlp_dropout"       : [0.3, 0.2, 0.0],
    "mlp_lr"            : 1e-3,
    "mlp_weight_decay"  : 1e-4,
    "mlp_epochs"        : 30,
    "mlp_patience"      : 5,
    "mlp_batch_size"    : 2048,    # training batch size

    # Inference
    "inference_batch_size": 8192,  # for scoring large test sets

    # Bootstrap CI
    "bootstrap_n"       : 1000,
    "bootstrap_ci"      : 0.95,

    # Recall sanity
    "recall_warn_threshold": 0.15,
}

# ============================================================
# 5. Feature Names (28 features, fixed order)
# ============================================================
# Features 0-19 are the original behavioural/lexical set.
# Features 20-27 are the Q1.2 within-session set. They are OPTIONAL: when no
# session context is supplied they are all 0.0, so a model may be trained with
# or without them. Both datasets are supported: EB-NeRD provides session_id;
# MIND does not, so those columns are constant zero there and are reported by
# src/serving_ablation.py's constant_columns() diagnostic.

SESSION_FEATURE_NAMES = [
    "session_impression_index",   # 0-based position of this impression in its session
    "session_prior_impressions",  # number of earlier impressions in the same session
    "session_prior_clicks",       # clicks accumulated in earlier session impressions
    "session_prior_ctr",          # prior clicks / prior inviews within the session
    "session_prior_read_time",    # mean dwell time (s) of earlier session impressions
    "session_prior_scroll",       # mean scroll % of earlier session impressions
    "session_candidate_overlap",  # frac of this slate already seen earlier in session
    "session_prior_cat_match",    # frac of prior session clicks in candidate's category
]

FEATURE_NAMES = [
    "bm25_score",
    "dense_score",
    "hybrid_score",
    "category_match",
    "category_affinity",
    "history_len",
    "recency_score",
    "avg_read_time_hist",
    "avg_scroll_hist",
    "click_overlap",
    "freshness_days",
    "freshness_log",
    "popularity_log",
    "inview_rate",
    "readtime_rate",
    "sentiment_score",
    "position_in_impression",
    "impression_size",
    "is_subscriber",
    "user_cat_entropy",
] + SESSION_FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)  # 28
N_BASE_FEATURES = 20             # original set, for backwards-compatible slicing

# Feature groups, used by the ablations to report metrics on subsets.
BASE_FEATURE_IDX = list(range(N_BASE_FEATURES))
SESSION_FEATURE_IDX = list(range(N_BASE_FEATURES, N_FEATURES))

SHARED_INDEX_CATALOG = True

TRAIN_SAMPLE_SIZE = 30_000

SERVING_UNAVAILABLE_FEATURES = [
    "popularity_log",    # idx 12 — log1p(total_pageviews), period aggregate
    "inview_rate",       # idx 13 — total_inviews / total_pageviews
    "readtime_rate",     # idx 14 — total_read_time / total_inviews
    "sentiment_score",   # idx 15 — editorial sentiment, post-hoc
]

SERVING_UNAVAILABLE_IDX = [
    FEATURE_NAMES.index(n) for n in SERVING_UNAVAILABLE_FEATURES
]

assert len(SERVING_UNAVAILABLE_IDX) == len(SERVING_UNAVAILABLE_FEATURES), (
    "SERVING_UNAVAILABLE_FEATURES must all exist in FEATURE_NAMES"
)

# ============================================================
# 6. MIND TSV Column Names (no header in file)
# ============================================================

MIND_NEWS_COLS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]

MIND_BEHAVIOR_COLS = [
    "impression_id", "user_id", "time", "history", "impressions",
]

# ============================================================
# 7. Validation (called at import time if run as main)
# ============================================================

def print_config():
    """Print path availability summary for debugging."""
    print(f"\n{'='*60}")
    print(f"  IRE A2 Config — Device: {DEVICE}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"{'='*60}")
    print("\n  EB-NeRD Paths:")
    for k in ["ebnerd_train_behaviors", "ebnerd_val_behaviors",
              "ebnerd_test_behaviors", "ebnerd_articles"]:
        p = PATHS[k]
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"    [{status}] {k}: {p}")
    print("\n  MIND Paths:")
    for k in ["mind_train_behaviors", "mind_val_behaviors", "mind_test_behaviors"]:
        p = PATHS[k]
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"    [{status}] {k}: {p}")
    print(f"\n  Features: {N_FEATURES} — {FEATURE_NAMES[:5]}...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_config()
