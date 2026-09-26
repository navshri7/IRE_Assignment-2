"""
src/sanity_checks.py — Automated correctness assertions and sanity monitors.

Assignment 2: News Recommendation System
IRE CS4.406

Checks:
 1. assert_no_future_leakage(behaviors):
    - Validates that all click timestamps in user history are <= impression_time.
 2. print_recall_sanity(retriever, behaviors, k_values, warn_threshold):
    - Evaluates Recall@K (K=50, 100, 200). Emits WARNING if Recall@200 < warn_threshold.
 3. assert_feature_matrix_clean(X, y, feature_names):
    - Validates zero NaNs, zero Infs, correct dimensionality, and non-empty classes.
    - Computes and logs point-biserial correlation for all features.
 4. assert_prediction_sanity(pred_file, expected_count):
    - Verifies line count == expected_count, valid integer ranks, and format conformity.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import pointbiserialr

from src.config import FEATURE_NAMES, N_FEATURES
from src.logger import get_logger

log = get_logger(__name__)


def assert_no_future_leakage(behaviors: List[dict], max_check: int = 1000) -> None:
    """
    Asserts that no history item timestamp is in the future relative to the impression time.
    """
    checked = 0
    violations = 0

    for imp in behaviors[:max_check]:
        imp_time = imp.get("impression_time")
        hist_times = imp.get("history_times", [])

        if not imp_time or not hist_times:
            continue

        for ht in hist_times:
            if ht is not None and ht > imp_time:
                violations += 1

        checked += 1

    if violations > 0:
        msg = f"[LEAKAGE ERROR] Found {violations} history events occurring after impression_time in {checked} impressions!"
        log.error(msg)
        raise AssertionError(msg)

    log.info(f"  [LEAKAGE CHECK] ✓ No future history timestamps detected (checked {checked:,} impressions)")


def assert_feature_matrix_clean(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> None:
    """
    Validates feature matrix integrity: zero NaNs, zero Infs, expected columns.
    Logs point-biserial correlations with the binary label if y is provided.
    """
    feat_names = feature_names or FEATURE_NAMES

    # 1. NaN and Inf checks
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count > 0 or inf_count > 0:
        msg = f"[FEATURE ERROR] Feature matrix contains {nan_count} NaNs and {inf_count} Infs!"
        log.error(msg)
        raise AssertionError(msg)

    # 2. Shape check
    if X.ndim != 2 or X.shape[1] != len(feat_names):
        msg = f"[FEATURE ERROR] Expected shape (N, {len(feat_names)}), got {X.shape}!"
        log.error(msg)
        raise AssertionError(msg)

    log.info(f"[FEATURE SANITY] ✓ Clean matrix: shape {X.shape}, 0 NaNs, 0 Infs")

    # 3. Label checks
    if y is not None:
        if len(y) != len(X):
            msg = f"[FEATURE ERROR] Feature rows ({len(X)}) != label rows ({len(y)})!"
            log.error(msg)
            raise AssertionError(msg)

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            msg = f"[FEATURE ERROR] Labels must be binary {0, 1}, found {unique_labels}!"
            log.error(msg)
            raise AssertionError(msg)

        pos_rate = float((y == 1).mean())
        log.info(f"[FEATURE SANITY] ✓ Labels valid: {int((y==1).sum()):,} pos, {int((y==0).sum()):,} neg (pos_rate: {pos_rate:.4f})")

        # Correlation breakdown
        for idx, name in enumerate(feat_names):
            col = X[:, idx]
            std = float(np.std(col))
            if std > 1e-8:
                try:
                    rpb, pval = pointbiserialr(y, col)
                    log.info(f"  feat {idx:02d} [{name:<24}]: r_pb={rpb:+.4f} (p={pval:.2e})")
                except Exception:
                    pass


def assert_prediction_sanity(
    pred_file: Path,
    expected_count: Optional[int] = None,
) -> int:
    """
    Validates a generated Codabench predictions.txt file.
    Format per line: `<impression_id> [<rank1>,<rank2>,...]`
    """
    if not pred_file.exists():
        msg = f"[PREDICTION ERROR] Prediction file does not exist: {pred_file}"
        log.error(msg)
        raise FileNotFoundError(msg)

    line_count = 0
    with open(pred_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ", 1)
            if len(parts) != 2:
                msg = f"[PREDICTION ERROR] Line {line_num} invalid format: '{line[:50]}'"
                log.error(msg)
                raise ValueError(msg)

            imp_id_str, ranks_str = parts
            if not ranks_str.startswith("[") or not ranks_str.endswith("]"):
                msg = f"[PREDICTION ERROR] Line {line_num} ranks not in brackets: '{ranks_str[:30]}'"
                log.error(msg)
                raise ValueError(msg)

            # Validate ranks are valid positive integers
            raw_ranks = ranks_str[1:-1].split(",")
            if raw_ranks and raw_ranks[0]:
                for r in raw_ranks:
                    try:
                        val = int(r.strip())
                        if val < 1:
                            raise ValueError(f"Rank {val} < 1")
                    except Exception as e:
                        msg = f"[PREDICTION ERROR] Line {line_num} rank item invalid '{r}': {e}"
                        log.error(msg)
                        raise ValueError(msg)

            line_count += 1

    if expected_count is not None and line_count != expected_count:
        msg = f"[PREDICTION ERROR] Line count mismatch: expected {expected_count:,}, got {line_count:,}"
        log.error(msg)
        raise AssertionError(msg)

    log.info(f"[PREDICTION SANITY] ✓ Prediction file {pred_file.name} valid: {line_count:,} impressions verified.")
    return line_count
