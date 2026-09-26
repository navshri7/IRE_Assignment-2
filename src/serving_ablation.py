"""

CLI USAGE:
  .venv/bin/python3 src/serving_ablation.py --dataset both --sample_size 500
  .venv/bin/python3 src/serving_ablation.py --dataset ebnerd --n_bootstrap 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import (
    FEATURE_NAMES,
    HYPERPARAMS,
    N_FEATURES,
    PATHS,
    SERVING_UNAVAILABLE_FEATURES,
    SERVING_UNAVAILABLE_IDX,
)
from src.data_loader import load_articles, load_behaviors
from src.evaluator import bootstrap_ci, mrr_score, ndcg_score
from src.feature_engineering import FeatureExtractor, extract_features_dataset
from src.logger import get_logger, log_section
from src.reranker import LightGBMReranker

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Column subsetting
# ─────────────────────────────────────────────────────────────────────────────

def serving_legal_idx() -> List[int]:
    """Column indices that ARE computable at serving time."""
    blocked = set(SERVING_UNAVAILABLE_IDX)
    return [i for i in range(N_FEATURES) if i not in blocked]


def _session_ctx(dataset: str, split: str) -> dict:
    """Within-session context for a split, or {} when unavailable (e.g. MIND)."""
    try:
        from run_all_experiments import build_session_context_for_split

        return build_session_context_for_split(dataset, split)
    except Exception as e:
        log.info(f"Session context unavailable for {dataset}.{split}: {e}")
        return {}


def constant_columns(X: np.ndarray) -> List[int]:
    """Indices of columns with zero variance in X (carry no signal)."""
    return [i for i in range(X.shape[1]) if float(np.std(X[:, i])) <= 1e-8]


def subset_columns(X: np.ndarray, keep: List[int]) -> np.ndarray:
    """Selects `keep` columns, preserving order."""
    return np.ascontiguousarray(X[:, keep])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-impression metric extraction (for paired bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def per_impression_metrics(
    behaviors,
    scores_per_arm: Dict[str, List[List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Computes AUC / MRR / nDCG@5 / nDCG@10 per impression for each arm.

    AUC is NaN on single-class impressions and excluded downstream (coercing it
    to 0.5 would bias the paired mean).
    """
    out: Dict[str, Dict[str, List[float]]] = {
        arm: {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
        for arm in scores_per_arm
    }

    for idx, imp in enumerate(behaviors):
        labels = imp.get("labels")
        if not labels or sum(labels) == 0 or sum(labels) == len(labels):
            continue
        for arm, all_scores in scores_per_arm.items():
            s = all_scores[idx]
            if len(s) != len(labels):
                continue
            try:
                auc = float(roc_auc_score(labels, s))
            except Exception:
                auc = float("nan")
            out[arm]["AUC"].append(auc)
            out[arm]["MRR"].append(mrr_score(labels, s))
            out[arm]["nDCG@5"].append(ndcg_score(labels, s, k=5))
            out[arm]["nDCG@10"].append(ndcg_score(labels, s, k=10))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. The ablation itself
# ─────────────────────────────────────────────────────────────────────────────

def run_serving_ablation(
    dataset: str,
    sample_size: int = 500,
    n_bootstrap: int = 1000,
    top_k: Optional[int] = None,
    n_estimators: int = 300,
    run_dir: Optional[Path] = None,
    save: bool = True,
) -> str:
    """
    Trains matched re-rankers on (a) all 20 features and (b) the 16
    serving-legal features, then reports both plus the paired delta.
    """
    log_section(log, f"Q9 SERVING-FEATURE ABLATION: {dataset.upper()}")

    articles = load_articles(dataset, split="val")
    session_ctx = _session_ctx(dataset, "val")

    log.info(f"Extracting features on [train] (sample={sample_size}) ...")
    train_behs = load_behaviors(dataset, split="train", sample_size=sample_size)
    Xtr, ytr, gtr, _, _ = extract_features_dataset(
        FeatureExtractor(articles, top_k=top_k, session_context=session_ctx),
        train_behs, show_progress=False,
    )

    log.info(f"Extracting features on [val] (sample={sample_size}) ...")
    val_behs = load_behaviors(dataset, split="val", sample_size=sample_size)
    val_behs = [b for b in val_behs if b.get("candidates")]
    Xva, yva, gva, _, _ = extract_features_dataset(
        FeatureExtractor(articles, top_k=top_k, session_context=session_ctx),
        val_behs, show_progress=False,
    )

    if len(Xva) == 0:
        log.error("No validation features extracted; aborting.")
        return ""

    # ── Diagnostics: which blocked columns are actually dead on this dataset ─
    dead = constant_columns(Xva)
    blocked_dead = [
        SERVING_UNAVAILABLE_FEATURES[SERVING_UNAVAILABLE_IDX.index(i)]
        for i in dead
        if i in SERVING_UNAVAILABLE_IDX
    ]
    log.info(f"Constant (zero-variance) columns on {dataset.upper()} val: "
             f"{[FEATURE_NAMES[i] for i in dead] or 'none'}")
    if blocked_dead:
        log.info(
            f"  -> {blocked_dead} are identically zero on {dataset.upper()}, so the two "
            "arms are expected to tie here. The ablation is only informative on datasets "
            "with real engagement counters (EB-NeRD)."
        )

    keep_idx = serving_legal_idx()
    log.info(
        f"Arm A (full)      : {N_FEATURES} features\n"
        f"Arm B (serving)   : {len(keep_idx)} features "
        f"(dropped: {SERVING_UNAVAILABLE_FEATURES})"
    )

    # ── Train both arms with identical hyperparameters ──────────────────────
    log.info("Training Arm A (full 20-feature re-ranker) ...")
    model_full = LightGBMReranker(
        n_estimators=n_estimators,
        num_leaves=HYPERPARAMS.get("lgbm_num_leaves", 63),
        learning_rate=HYPERPARAMS.get("lgbm_learning_rate", 0.05),
    )
    model_full.fit(Xtr, ytr, gtr)

    log.info(f"Training Arm B ({len(keep_idx)}-feature serving-legal re-ranker) ...")
    Xtr_b = subset_columns(Xtr, keep_idx)
    Xva_b = subset_columns(Xva, keep_idx)
    model_serving = LightGBMReranker(
        n_estimators=n_estimators,
        num_leaves=HYPERPARAMS.get("lgbm_num_leaves", 63),
        learning_rate=HYPERPARAMS.get("lgbm_learning_rate", 0.05),
    )
    model_serving.fit(Xtr_b, ytr, gtr)

    # ── Score the validation set with both arms ─────────────────────────────
    scores_full = model_full.predict_scores(Xva)
    scores_serving = model_serving.predict_scores(Xva_b)

    # Unpack flat scores back into per-impression lists
    def _unpack(flat: np.ndarray, groups: np.ndarray) -> List[List[float]]:
        out, i = [], 0
        for n in groups:
            out.append(flat[i : i + n].tolist())
            i += n
        return out

    arms = {
        "full": _unpack(scores_full, gva),
        "serving": _unpack(scores_serving, gva),
    }

    per_imp = per_impression_metrics(val_behs, arms)

    # ── Report ──────────────────────────────────────────────────────────────
    lines = [
        f"# Q9 Serving-Feature Ablation: {dataset.upper()}",
        f"**Impressions**: {len(per_imp['full']['MRR']):,} | "
        f"**Bootstrap**: {n_bootstrap:,} | **Seed**: 42\n",
        "## Dropped Columns (not observable at serving time)",
        "| Feature | Definition | Why unavailable at serving time |",
        "|:--|:--|:--|",
        f"| `popularity_log` | log1p(total_pageviews) | Period aggregate over the whole evaluation window |",
        f"| `inview_rate` | total_inviews / total_pageviews | Period aggregate |",
        f"| `readtime_rate` | total_read_time / total_inviews | Period aggregate |",
        f"| `sentiment_score` | editorial sentiment | Post-hoc editorial judgement |",
        "",
        "## Metrics: Full vs. Serving-Legal Features (95% CI)",
        "| Arm | #Features | AUC | MRR | nDCG@5 | nDCG@10 |",
        "|:--|--:|:---:|:---:|:------:|:-------:|",
    ]

    for arm, label, nfeat in [
        ("full", "A: Full (headline)", N_FEATURES),
        ("serving", "B: Serving-legal only", len(keep_idx)),
    ]:
        cells = []
        for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
            vals = [v for v in per_imp[arm][m] if not np.isnan(v)]
            mean, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
            cells.append(f"{mean:.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"| {label} | {nfeat} | " + " | ".join(cells) + " |")

    # Paired delta: A - B, resampled over impressions
    lines += [
        "",
        "## Paired Delta (A: Full - B: Serving-legal), 95% Bootstrap CI",
        "| Metric | Delta | 95% CI | Excludes 0? |",
        "|:--|--:|:--|:--:|",
    ]
    deltas_report = []
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        a = per_imp["full"][m]
        b = per_imp["serving"][m]
        pair = [
            x - y for x, y in zip(a, b) if not np.isnan(x) and not np.isnan(y)
        ]
        if not pair:
            continue
        mean, lo, hi = bootstrap_ci(pair, n_bootstrap=n_bootstrap)
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {m} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {excl} |")
        deltas_report.append((m, mean, lo, hi, excl))

    # LightGBM gain attribution for the dropped columns
    imp_full = model_full.get_feature_importances()
    imp_serv = model_serving.get_feature_importances()
    blocked = set(SERVING_UNAVAILABLE_FEATURES)
    blocked_gain = sum(v for k, v in imp_full.items() if k in blocked)
    total_gain = sum(imp_full.values()) or 1.0
    lines += [
        "",
        "## LightGBM Gain Attribution (Arm A)",
        f"- Total gain on dropped columns: **{blocked_gain:.2f} / {total_gain:.2f} "
        f"({100 * blocked_gain / total_gain:.1f}%)**",
        f"- Total gain on serving-legal columns: {100 * (1 - blocked_gain / total_gain):.1f}%",
        "",
        "### Per-feature gain (Arm A, top 10)",
        "| Feature | Gain | Serving-legal? |",
        "|:--|--:|:--:|",
    ]
    for name, val in sorted(imp_full.items(), key=lambda x: -x[1])[:10]:
        mark = "NO" if name in blocked else "yes"
        lines.append(f"| `{name}` | {val:.2f} | {mark} |")

    report = "\n".join(lines)
    print("\n" + report + "\n")
    log.info("\n" + report)

    if save:
        if run_dir is not None:
            out = Path(run_dir) / f"serving_ablation_{dataset}.md"
        else:
            from src.run_dir import resolve_run_dir

            out = resolve_run_dir() / f"serving_ablation_{dataset}.md"
        out.write_text(report, encoding="utf-8")
        log.info(f"Saved to {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Q9: serving-feature-subset ablation")
    parser.add_argument("--dataset", choices=["ebnerd", "mind", "both"], default="both")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=None,
                        help="Stage-1 top-K pool (default None = re-rank inview slate)")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run folder for the report. Default: a fresh "
             "results/runs/<YYYYmmdd_HHMMSS>/ folder.",
    )
    args = parser.parse_args()

    from src.run_dir import resolve_run_dir

    run_dir = resolve_run_dir(args.run_dir)
    log.info(f"Run directory: {run_dir}")

    datasets = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_serving_ablation(
            dataset=ds,
            sample_size=args.sample_size,
            n_bootstrap=args.n_bootstrap,
            top_k=args.top_k,
            n_estimators=args.n_estimators,
            run_dir=run_dir,
        )


if __name__ == "__main__":
    main()
