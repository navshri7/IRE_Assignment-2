"""
src/predictor.py — Chunked batch predictor and Codabench submission generator.

MODEL TYPES:
  bm25     – BM25 sparse retriever
  dense    – TF-IDF + SVD dense retriever
  hybrid   – Linear combination of BM25 + Dense
  lgbm     – LightGBM reranker (trained on features)
  mlp      – PyTorch MLP reranker (trained on features)
  official – Unified official popularity baseline (ebnerd-benchmark formulation)
  q3       – Category-Aware Freshness Ranker (Q3 principled improvement over official)

CLI USAGE:
  .venv/bin/python3 src/predictor.py --dataset ebnerd --model lgbm     --split val --sample_size 500
  .venv/bin/python3 src/predictor.py --dataset mind   --model mlp      --split val --sample_size 500
  .venv/bin/python3 src/predictor.py --dataset ebnerd --model official  --split test
  .venv/bin/python3 src/predictor.py --dataset mind   --model q3        --split test
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from src.baseline_official import CategoryAwareFreshnessRanker, OfficialBaselineRanker
from src.config import HYPERPARAMS, PATHS
from src.data_loader import (
    ArticleDict,
    ImpressionList,
    load_articles,
    load_behaviors,
    stream_behaviors,
)
from src.evaluator import scores_to_ranks, write_codabench_predictions
from src.feature_engineering import FeatureExtractor, extract_features_dataset
from src.logger import get_logger
from src.reranker import LightGBMReranker, MLPReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_dense_retriever
from src.sanity_checks import assert_prediction_sanity

log = get_logger(__name__)

# Canonical filename each Codabench portal expects INSIDE the submitted zip.
CANONICAL_ARCNAME = {
    # RecSys 2024 Challenge: default in ebrec.utils.write_submission_file and
    # used by every ebnerd-benchmark example script.
    "ebnerd": "predictions.txt",
    # MIND: the official sample_pred/prediction.txt read by MIND's evaluate.py.
    "mind": "prediction.txt",
}


def generate_submission_for_batch(
    impressions_batch: ImpressionList,
    model_type: str,
    model_obj,
    extractor: Optional[FeatureExtractor] = None,
) -> List[Tuple[int, List[int]]]:
    """
    Scores candidates for a batch of impressions and converts scores to 1-indexed ranks.

    Returns:
        List of (impression_id, [rank1, rank2, ...])
    """
    results: List[Tuple[int, List[int]]] = []

    if model_type in ["bm25", "dense", "hybrid", "official"]:
        # All retrieval-style models expose the same rank_impression(history, candidates) API
        ranker = model_obj
        for imp in impressions_batch:
            imp_id = imp["impression_id"]
            cands = imp.get("candidates", [])
            if not cands:
                results.append((imp_id, []))
                continue

            scores = ranker.rank_impression(imp.get("history", []), cands)
            ranks = scores_to_ranks(scores)
            results.append((imp_id, ranks))

    elif model_type == "q3":
        # CategoryAwareFreshnessRanker uses score_impression(imp_dict) to access timestamp
        ranker = model_obj
        for imp in impressions_batch:
            imp_id = imp["impression_id"]
            cands = imp.get("candidates", [])
            if not cands:
                results.append((imp_id, []))
                continue

            scores = ranker.score_impression(imp)
            ranks = scores_to_ranks(scores)
            results.append((imp_id, ranks))

    elif model_type in ["lgbm", "mlp"]:
        reranker = model_obj
        X, _, groups, imp_ids, cand_ids = extract_features_dataset(
            extractor, impressions_batch, show_progress=False
        )
        if len(X) == 0:
            for imp in impressions_batch:
                results.append((imp["impression_id"], []))
            return results

        flat_scores = reranker.predict_scores(X)

        idx = 0
        for imp in impressions_batch:
            imp_id = imp["impression_id"]
            n = len(imp.get("candidates", []))
            if n == 0:
                results.append((imp_id, []))
            else:
                imp_scores = flat_scores[idx : idx + n].tolist()
                ranks = scores_to_ranks(imp_scores)
                results.append((imp_id, ranks))
                idx += n
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return results


def run_predictor(
    dataset: str,
    model_type: str,
    split: str = "test",
    sample_size: Optional[int] = None,
    output_filename: Optional[str] = None,
    batch_size: int = 25_000,
    top_k: Optional[int] = None,
    max_history: Optional[int] = None,
    expected_total: Optional[int] = None,
    progress_every_s: float = 60.0,
) -> Tuple[Path, Path]:
    """
    End-to-end submission generation pipeline with memory-safe batch streaming.

    top_k:
        Stage-1 top-K candidate generation. MUST stay None for split='test' —
        Codabench requires a permutation of the impression's inview candidates,
        and FeatureExtractor.assert_retrieval_mode_allowed() enforces this.

    max_history:
        Cap on retained clicks per user. Required to keep the 13.5M-row EB-NeRD
        test set traversable (its history averages 144.6 clicks/user).

    expected_total:
        Row count of the behaviours file, used only to render a % and ETA in the
        progress heartbeat.

    Returns:
        (txt_path, zip_path)
    """
    # Guard against invalid submissions (retrieved articles outside the inview set).
    FeatureExtractor.assert_retrieval_mode_allowed(top_k, split)

    if expected_total is None and split == "test":
        try:
            import polars as pl

            expected_total = (
                pl.scan_parquet(PATHS[f"ebnerd_{split}_behaviors"])
                .select(pl.len())
                .collect()
                .item()
            )
        except Exception:
            expected_total = None

    sub_dir = PATHS["submissions"]
    sub_dir.mkdir(parents=True, exist_ok=True)

    base_name = output_filename or f"prediction_{dataset}_{model_type}_{split}"
    txt_path = sub_dir / f"{base_name}.txt"
    zip_path = sub_dir / f"{base_name}.zip"

    log.info("=" * 60)
    log.info(f"  Phase 6: Predictor — {dataset.upper()} [{model_type.upper()}] on [{split}]")
    log.info(f"  Target TXT : {txt_path}")
    log.info(f"  Target ZIP : {zip_path}")
    log.info("=" * 60)

    # 1. Load articles for the target split
    articles = load_articles(dataset, split=split)

    # 2. Prepare model / retriever / extractor
    extractor = None
    if model_type == "bm25":
        model_obj = BM25Retriever(articles)
    elif model_type == "dense":
        model_obj = build_dense_retriever(articles)
    elif model_type == "hybrid":
        bm25 = BM25Retriever(articles)
        dense = build_dense_retriever(articles)
        model_obj = HybridRetriever(bm25, dense)
    elif model_type == "official":
        # Unified official popularity baseline (ebnerd-benchmark formulation)
        # For MIND, we load training behaviors to compute click/inview counts
        train_behaviors = None
        if dataset == "mind":
            try:
                log.info("Loading MIND train behaviors to compute popularity stats ...")
                train_behaviors = load_behaviors(dataset, split="train", sample_size=50_000)
            except Exception as e:
                log.warning(f"Could not load MIND train behaviors: {e}")
        model_obj = OfficialBaselineRanker(
            dataset=dataset,
            articles=articles,
            train_behaviors=train_behaviors,
        )
    elif model_type == "q3":
        # Category-Aware Freshness Ranker: built on top of the official baseline
        train_behaviors = None
        if dataset == "mind":
            try:
                log.info("Loading MIND train behaviors for Q3 popularity stats ...")
                train_behaviors = load_behaviors(dataset, split="train", sample_size=50_000)
            except Exception as e:
                log.warning(f"Could not load MIND train behaviors: {e}")
        base_ranker = OfficialBaselineRanker(
            dataset=dataset,
            articles=articles,
            train_behaviors=train_behaviors,
        )
        model_obj = CategoryAwareFreshnessRanker(
            base_ranker=base_ranker,
            articles=articles,
            mode="full",
        )
    elif model_type in ("lgbm", "mlp"):
        # Delegate to the shared load-or-retrain helper so a checkpoint left
        # over from an older FEATURE_NAMES (e.g. the 20-column pre-session
        # matrix) is discarded and retrained instead of crashing the run.
        from run_all_experiments import load_or_retrain_reranker

        model_obj, retrained = load_or_retrain_reranker(
            dataset, model_type, train_sample=2000, session_ctx={}
        )
        if retrained:
            log.warning(
                f"[{model_type}] No usable checkpoint for {dataset}; trained a fresh "
                f"model on the [train] split for this run."
            )
        extractor = FeatureExtractor(articles, top_k=top_k)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 3. Stream behaviors, predict ranks in batches, and write to txt
    total_written = 0
    t_start = time.perf_counter()
    last_heartbeat = t_start
    with open(txt_path, "w", encoding="utf-8") as f_out:
        if sample_size is not None:
            # Sliced debug mode
            behaviors = load_behaviors(dataset, split=split, sample_size=sample_size)
            batch_results = generate_submission_for_batch(
                behaviors, model_type, model_obj, extractor=extractor
            )
            for imp_id, ranks in batch_results:
                f_out.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
                total_written += 1
        else:
            # Full streaming mode
            stream = stream_behaviors(
                dataset, split=split, batch_size=batch_size, max_history=max_history
            )
            for batch in tqdm(stream, desc=f"Predicting & Streaming ({dataset})"):
                batch_results = generate_submission_for_batch(
                    batch, model_type, model_obj, extractor=extractor
                )
                for imp_id, ranks in batch_results:
                    f_out.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
                    total_written += 1

                # Heartbeat: rate + ETA, greppable from the nohup log.
                now = time.perf_counter()
                if now - last_heartbeat >= progress_every_s:
                    last_heartbeat = now
                    elapsed = now - t_start
                    rate = total_written / max(elapsed, 1e-9)
                    if expected_total:
                        eta_h = (expected_total - total_written) / max(rate, 1e-9) / 3600.0
                        log.info(
                            f"[PROGRESS] {total_written:,}/{expected_total:,} "
                            f"({100.0 * total_written / expected_total:.1f}%) "
                            f"{rate:.0f} imp/s | elapsed {elapsed / 60:.1f}m "
                            f"| ETA {eta_h:.2f}h"
                        )
                    else:
                        log.info(
                            f"[PROGRESS] {total_written:,} impressions | "
                            f"{rate:.0f} imp/s | elapsed {elapsed / 60:.1f}m"
                        )
                    f_out.flush()

    log.info(
        f"Wrote {total_written:,} predictions to {txt_path} "
        f"in {(time.perf_counter() - t_start) / 60:.1f} min"
    )

    # 4. Create ZIP archive.
    #
    # Codabench matches the file INSIDE the archive by name, so the arcname
    # must be the canonical name, not our descriptive one:
    #   RecSys 2024 (EB-NeRD) -> predictions.txt
    #     (ebnerd-benchmark/src/ebrec/utils/_python.py:65 default, and every
    #      examples/* script writes path=".../predictions.txt")
    #   MIND                   -> prediction.txt
    #     (msnews/MIND sample_pred/prediction.txt, the file the official
    #      evaluate.py reads)
    # Zipping prediction_ebnerd_lgbm_test.txt is very likely rejected, so we
    # keep the readable .txt on disk and rename it inside the archive.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=CANONICAL_ARCNAME.get(dataset, "predictions.txt"))
    log.info(
        f"Created submission ZIP: {zip_path} "
        f"({zip_path.stat().st_size / (1024*1024):.2f} MB, "
        f"inner name '{CANONICAL_ARCNAME.get(dataset)}')"
    )

    # 5. Sanity assertion
    assert_prediction_sanity(txt_path, expected_count=total_written)
    log.info("Phase 6 predictor pipeline complete.")

    return txt_path, zip_path


generate_submission_file = run_predictor


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Submission Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model choices:\n"
            "  bm25     – BM25 sparse retriever\n"
            "  dense    – TF-IDF + SVD dense retriever\n"
            "  hybrid   – BM25 + Dense linear combination\n"
            "  lgbm     – LightGBM reranker (requires trained model)\n"
            "  mlp      – PyTorch MLP reranker (requires trained model)\n"
            "  official – Unified official popularity baseline (Q3 reference)\n"
            "  q3       – Category-Aware Freshness Ranker (Q3 improvement)\n"
        ),
    )
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="ebnerd", help="Dataset")
    parser.add_argument(
        "--model",
        choices=["bm25", "dense", "hybrid", "lgbm", "mlp", "official", "q3"],
        default="lgbm",
        help="Model type (see epilog for details)",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split (val for debug, test for final)")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size cap (None for full set)")
    parser.add_argument("--batch_size", type=int, default=25_000, help="Streaming batch size")
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=(
            "Stage-1 top-K candidate generation (val split only). "
            "Omit to re-rank the impression's inview candidates. "
            "Rejected for --split test because Codabench requires ranking "
            "exactly the inview candidate set."
        ),
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=None,
        help="Cap on retained clicks per user (default: HYPERPARAMS['max_history_len'] = 50). "
             "Required to keep the 13.5M-row EB-NeRD test set in memory.",
    )
    parser.add_argument(
        "--progress_every_s",
        type=float,
        default=60.0,
        help="Seconds between progress heartbeats (rate + ETA) in the log",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Override output basename (default: prediction_<dataset>_<model>_<split>)",
    )
    args = parser.parse_args()

    run_predictor(
        dataset=args.dataset,
        model_type=args.model,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_history=args.max_history,
        progress_every_s=args.progress_every_s,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
