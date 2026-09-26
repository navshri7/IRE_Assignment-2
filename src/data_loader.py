"""
src/data_loader.py — Unified data loading for EB-NeRD and MIND datasets.

Assignment 2: News Recommendation System
IRE CS4.406

REUSES (from ire-assignment1/data_utils.py):
  - load_news()              → extended to unified article schema
  - load_behaviors()         → extended with rich history + session fields
  - load_ebnerd_articles()   → same logic, extended schema
  - load_ebnerd_history()    → same logic, now returns full history schema
  - stream_ebnerd_behaviors()→ extended with session fields

EXTENDS WITH:
  - Rich history schema: history_times, history_read_times, history_scroll
  - MIND: ignore_errors=True (abstracts contain unescaped quotes in TSV)
  - stream_behaviors_batched(): memory-safe generator for 13.5M test rows
  - --sample_size CLI for debug runs
  - Unified article dict schema for both datasets
  - load_articles(): router for both datasets

CLI USAGE:
  .venv/bin/python3 src/data_loader.py --dataset ebnerd --split val --sample_size 100
  .venv/bin/python3 src/data_loader.py --dataset mind --split train --sample_size 200
"""

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.config import PATHS, MIND_NEWS_COLS, MIND_BEHAVIOR_COLS, HYPERPARAMS
from src.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified Type Aliases
# ─────────────────────────────────────────────────────────────────────────────

ArticleDict = Dict[str, dict]   # article_id (str) → article metadata dict
ImpressionList = List[dict]     # list of unified impression dicts

# ─────────────────────────────────────────────────────────────────────────────
# 2. Article Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_ebnerd_articles(articles_path: Optional[Path] = None) -> ArticleDict:
    """
    Load EB-NeRD articles.parquet into a unified article dict.
    REUSES: ire-assignment1/data_utils.py::load_ebnerd_articles() logic.
    EXTENDS: adds published_time, total_pageviews, total_inviews,
             total_read_time, sentiment_score.

    Returns:
        dict: article_id (str) → {title, abstract, category, published_time,
              total_pageviews, total_inviews, total_read_time,
              sentiment_score, text}
    """
    path = articles_path or PATHS["ebnerd_articles"]
    log.info(f"Loading EB-NeRD articles from {path}")

    df = pl.read_parquet(path).select([
        pl.col("article_id").cast(pl.Utf8).alias("article_id"),
        pl.col("title").fill_null("").alias("title"),
        pl.col("subtitle").fill_null("").alias("abstract"),
        pl.col("category_str").fill_null("unknown").alias("category"),
        pl.col("body").fill_null("").str.slice(0, 2000).alias("text_body"),
        pl.col("published_time"),
        pl.col("total_pageviews").fill_null(0).cast(pl.Int32).alias("total_pageviews"),
        pl.col("total_inviews").fill_null(0).cast(pl.Int32).alias("total_inviews"),
        pl.col("total_read_time").fill_null(0.0).alias("total_read_time"),
        pl.col("sentiment_score").fill_null(0.0).alias("sentiment_score"),
    ])

    articles: ArticleDict = {}
    for row in df.iter_rows(named=True):
        aid = row["article_id"]
        t, a = row["title"], row["abstract"]
        articles[aid] = {
            "article_id"     : aid,
            "title"          : t,
            "abstract"       : a,
            "text_body"      : row["text_body"],
            "category"       : row["category"],
            "published_time" : row["published_time"],    # datetime | None
            "total_pageviews": row["total_pageviews"],
            "total_inviews"  : row["total_inviews"],
            "total_read_time": row["total_read_time"],
            "sentiment_score": row["sentiment_score"],
            "text"           : f"{t} {a}".strip(),
        }

    log.info(f"  Loaded {len(articles):,} EB-NeRD articles")
    return articles


def load_mind_articles(news_path: Optional[Path] = None) -> ArticleDict:
    """
    Load MIND news.tsv into a unified article dict.
    REUSES: ire-assignment1/data_utils.py::load_news() logic.
    EXTENDS: adds missing EB-NeRD-style fields as defaults.

    NOTE: ignore_errors=True required — abstracts contain unescaped quotes.
    """
    path = news_path or PATHS["mind_train_news"]
    log.info(f"Loading MIND articles from {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=MIND_NEWS_COLS,
        quote_char=None,
        ignore_errors=True,
    ).fill_null("")

    articles: ArticleDict = {}
    for row in df.iter_rows(named=True):
        nid = row["news_id"]
        t, a = row["title"], row["abstract"]
        articles[nid] = {
            "article_id"     : nid,
            "title"          : t,
            "abstract"       : a,
            "category"       : row["category"],
            "published_time" : None,   # not available in MIND
            "total_pageviews": 0,
            "total_inviews"  : 0,
            "total_read_time": 0.0,
            "sentiment_score": 0.0,
            "text"           : f"{t} {a}".strip(),
        }

    log.info(f"  Loaded {len(articles):,} MIND articles")
    return articles


def load_articles(
    dataset: str,
    split: str = "train",
    merge_splits: Optional[List[str]] = None,
) -> ArticleDict:
    """
    Router: load articles for either 'ebnerd' or 'mind'.

    merge_splits:
        Merge the article catalogues of several splits into ONE index, with
        `split` taking precedence on conflict (so a later split overrides an
        earlier one field-by-field).

        WHY THIS EXISTS. MIND ships a different news.tsv per split:
        MINDsmall_train 51,282 articles, MINDsmall_dev 42,416, and only 28,460
        IDs are shared. Building the retrieval index per-split therefore makes
        the FEATURES themselves split-dependent:

          * BM25 IDF is a function of the document collection, so bm25_score
            shifts from mean 6.98 (train catalogue) to 8.53 (dev catalogue),
            correlation 0.81 between the two.
          * The mean-pooled user vector silently DROPS history articles that are
            absent from the catalogue, so dense_score shifts 0.115 -> 0.140.

        A re-ranker trained on one space and evaluated in another learns split
        points on train-time values and applies them to shifted values. Measured
        cost: 45.7% of the model's gain sat on features that moved between
        training and evaluation.

        Merging gives one index shared by training, evaluation and serving, so
        a feature means the same thing everywhere.

        EB-NeRD is unaffected: load_ebnerd_articles() returns the same
        articles.parquet for train and val, so merging is a no-op there.
    """
    if dataset == "ebnerd":
        if split == "test":
            return load_ebnerd_articles(PATHS["ebnerd_test_articles"])
        return load_ebnerd_articles()

    if dataset != "mind":
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'ebnerd' or 'mind'.")

    if not merge_splits:
        if split == "test":
            return load_mind_articles(PATHS["mind_test_news"])
        if split == "val":
            return load_mind_articles(PATHS["mind_val_news"])
        return load_mind_articles()

    # Merge, lowest precedence first so `split` wins.
    others = [s for s in merge_splits if s != split]
    ordered = others[::-1] + [split]
    merged: ArticleDict = {}
    for s in ordered:
        part = load_articles("mind", split=s)
        for aid, meta in part.items():
            if aid in merged:
                merged[aid] = {**merged[aid], **meta}
            else:
                merged[aid] = meta
    log.info(
        f"  Merged MIND article catalogue over splits {ordered}: {len(merged):,} articles"
    )
    return merged


def index_catalog_splits(dataset: str, split: str) -> Optional[List[str]]:
    """
    """
    if dataset != "mind":
        return None
    if split == "test":
        return ["train", "val", "test"]
    # Both "train" and "val" resolve to the same shared catalogue. This is the
    # load-bearing line: returning None for "train" reintroduces the skew.
    return ["train", "val"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EB-NeRD History Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_ebnerd_history_df(history_path: Path) -> pl.DataFrame:
    """Internal: load history.parquet as-is for merge."""
    if not history_path.exists():
        log.warning(f"History file not found: {history_path} — using empty history")
        return pl.DataFrame({"user_id": [], "article_id_fixed": [],
                             "impression_time_fixed": [], "read_time_fixed": [],
                             "scroll_percentage_fixed": []})
    return pl.read_parquet(history_path)


def _make_history_dict(
    hist_df: pl.DataFrame,
    max_history: Optional[int] = None,
) -> Dict[int, dict]:
    """
    Build user_id → {history, history_times, history_read_times, history_scroll} dict.
    REUSES: ire-assignment1/data_utils.py::load_ebnerd_history() logic.
    EXTENDS: includes timestamps + read_times + scroll for decay features.

    SCALING (critical for the 13.5M-row test set):
        EB-NeRD test history averages 144.6 clicks per user across 807,677 users
        — 116.8M history entries. Materialising all of them as Python strings
        takes many GB and minutes of wall clock, which makes the full test set
        untraversable.

        Two optimisations make it tractable:
          1. Truncation to the most recent `max_history` clicks, done VECTORISED
             in Polars via `list.tail()` rather than per-row Python slicing.
             This matches how the features actually consume history
             (bm25 uses the last 10 titles, the dense retriever the last 20, and
             history_len is already clipped at HYPERPARAMS['max_history_len']).
          2. `article_id_fixed` is List(Int32); the cast to Utf8 is done
             vectorised instead of a per-element `str()` call.

        Combined: ~40s instead of many minutes, at a documented truncation.
    """
    if hist_df.height == 0:
        return {}

    if max_history is None:
        max_history = HYPERPARAMS.get("max_history_len", 50)

    cols = [
        "user_id",
        "article_id_fixed",
        "impression_time_fixed",
        "read_time_fixed",
        "scroll_percentage_fixed",
    ]
    missing = [c for c in cols if c not in hist_df.columns]
    if missing:
        log.warning(f"History frame missing {missing}; returning empty history.")
        return {}

    exprs = []
    for c, dtype in [
        ("article_id_fixed", pl.List(pl.Utf8)),
        ("impression_time_fixed", None),
        ("read_time_fixed", None),
        ("scroll_percentage_fixed", None),
    ]:
        col = pl.col(c)
        if max_history:
            col = col.list.tail(max_history)
        if dtype is not None:
            col = col.cast(dtype)
        exprs.append(col.alias(c))

    prepared = hist_df.select([pl.col("user_id")] + exprs)

    history_dict: Dict[int, dict] = {}
    for uid, aids, times, reads, scrolls in prepared.iter_rows():
        history_dict[uid] = {
            "history":           aids,
            "history_times":     times,
            "history_read_times": reads,
            "history_scroll":    scrolls,
        }
    log.info(
        f"  History dict built for {len(history_dict):,} users "
        f"(capped at last {max_history} clicks/user)"
    )
    return history_dict


def build_session_context(
    behaviors_df: pl.DataFrame,
    articles: Optional[ArticleDict] = None,
    max_prior: int = 10,
    max_seen_articles: int = 300,
) -> Dict[int, dict]:
    required = {"session_id", "impression_id", "impression_time"}
    if not required.issubset(set(behaviors_df.columns)):
        log.warning(
            f"build_session_context: missing {sorted(required - set(behaviors_df.columns))}; "
            "returning empty context (session features will be 0.0)."
        )
        return {}

    has_clicked = "article_ids_clicked" in behaviors_df.columns
    df = behaviors_df.select(
        [
            c
            for c in [
                "session_id", "impression_id", "impression_time",
                "article_ids_inview", "article_ids_clicked",
                "read_time", "scroll_percentage",
            ]
            if c in behaviors_df.columns
        ]
    ).filter(pl.col("session_id").is_not_null())

    if df.height == 0:
        return {}

    # Deterministic within-session order.
    df = df.sort(["session_id", "impression_time", "impression_id"])

    context: Dict[int, dict] = {}
    # Running state for the current session.
    cur_sid = None
    idx = 0
    prior_imps = 0
    prior_clicks = 0
    prior_inviews = 0
    rt_list: List[float] = []
    sc_list: List[float] = []
    seen: set = set()
    seen_order: List[str] = []
    click_cats: Counter = Counter()

    def _emit(imp_id, i, pi, pc, pv, rts, scs, sn, cats):
        context[imp_id] = {
            "index": i,
            "prior_impressions": pi,
            "prior_clicks": pc,
            "prior_inviews": pv,
            "prior_read_time_mean": (sum(rts) / len(rts)) if rts else 0.0,
            "prior_scroll_mean": (sum(scs) / len(scs)) if scs else 0.0,
            "seen_articles": list(sn),
            "prior_click_categories": cats,
        }

    for row in df.iter_rows(named=False):
        (
            sid, imp_id, imp_time, inview, clicked, read_time, scroll,
        ) = list(row) + [None] * (7 - len(row))

        if sid != cur_sid:
            cur_sid = sid
            idx = 0
            prior_imps = 0
            prior_clicks = 0
            prior_inviews = 0
            rt_list.clear()
            sc_list.clear()
            seen.clear()
            seen_order.clear()
            click_cats.clear()

        # Emit the context for THIS impression from the prefix only.
        window_seen = seen_order[-max_seen_articles:]
        _emit(
            imp_id, idx, prior_imps, prior_clicks, prior_inviews,
            rt_list[-max_prior:], sc_list[-max_prior:], window_seen, dict(click_cats),
        )

        # ── Then fold this impression into the running state ────────────────
        inview_list = [str(a) for a in (inview or [])]
        clicked_list = [str(a) for a in (clicked or [])] if has_clicked else []

        for a in inview_list:
            if a not in seen:
                seen.add(a)
                seen_order.append(a)
        if len(seen_order) > max_seen_articles:
            drop = seen_order[:-max_seen_articles]
            seen_order = seen_order[-max_seen_articles:]
            for a in drop:
                seen.discard(a)

        # Category tally of clicks already spent in this session. Needs the
        # article catalog to resolve article_id -> category; without it the
        # session_prior_cat_match feature stays 0.0.
        if articles:
            for a in clicked_list:
                cat = (articles.get(a) or {}).get("category") or None
                if cat:
                    click_cats[cat] += 1

        prior_clicks += len(clicked_list)
        prior_inviews += len(inview_list)
        if read_time is not None and read_time > 0:
            rt_list.append(float(read_time))
        if scroll is not None and scroll > 0:
            sc_list.append(float(scroll))

        idx += 1
        prior_imps += 1

    log.info(
        f"  Session context built for {len(context):,} impressions "
        f"({df.height:,} session rows, max_prior={max_prior})"
    )
    return context


# ─────────────────────────────────────────────────────────────────────────────
# 4. EB-NeRD Behavior Streaming (memory-safe for 13.5M test rows)
# ─────────────────────────────────────────────────────────────────────────────

def stream_ebnerd_behaviors(
    behaviors_path: Path,
    history_dict: Dict[int, dict],
    batch_size: int = 50_000,
    max_rows: Optional[int] = None,
) -> Iterator[ImpressionList]:
    """
    Memory-safe streaming generator for EB-NeRD behaviors.parquet.
    REUSES: ire-assignment1/data_utils.py::stream_ebnerd_behaviors() logic.
    EXTENDS: rich impression schema with session + history_times fields.

    Yields:
        List[dict]: batch of unified impression dicts
    """
    beh_lazy = pl.scan_parquet(behaviors_path)
    total_rows = beh_lazy.select(pl.len()).collect().item()
    if max_rows:
        total_rows = min(total_rows, max_rows)

    schema_cols = beh_lazy.collect_schema().names()
    has_labels  = "article_ids_clicked" in schema_cols

    select_cols = [
        "impression_id", "user_id", "impression_time",
        "article_ids_inview",
    ]
    if has_labels:
        select_cols.append("article_ids_clicked")
    for opt_col in ["read_time", "scroll_percentage", "device_type",
                    "is_subscriber", "age", "gender", "session_id"]:
        if opt_col in schema_cols:
            select_cols.append(opt_col)

    log.info(f"  Streaming {total_rows:,} rows from {behaviors_path.name} "
             f"(batch={batch_size:,}, has_labels={has_labels})")

    for offset in range(0, total_rows, batch_size):
        chunk = min(batch_size, total_rows - offset)
        batch_df = (
            beh_lazy
            .slice(offset, chunk)
            .select(select_cols)
            .collect()
        )

        records: ImpressionList = []
        for row in batch_df.iter_rows(named=True):
            uid = row["user_id"]
            h   = history_dict.get(uid, {})

            inview = [str(a) for a in (row["article_ids_inview"] or [])]

            labels = None
            if has_labels:
                clicked_set = set(str(a) for a in (row.get("article_ids_clicked") or []))
                labels = [1 if aid in clicked_set else 0 for aid in inview]

            records.append({
                "impression_id"     : row["impression_id"],
                "user_id"           : str(uid),
                "impression_time"   : row.get("impression_time"),
                "history"           : h.get("history", []),
                "history_times"     : h.get("history_times", []),
                "history_read_times": h.get("history_read_times", []),
                "history_scroll"    : h.get("history_scroll", []),
                "candidates"        : inview,
                "labels"            : labels,
                "session_id"        : str(row["session_id"]) if row.get("session_id") else None,
                "read_time"         : row.get("read_time"),
                "scroll_percentage" : row.get("scroll_percentage"),
                "device_type"       : row.get("device_type"),
                "is_subscriber"     : row.get("is_subscriber"),
                "age"               : row.get("age"),
                "gender"            : row.get("gender"),
            })

        yield records


def load_ebnerd_behaviors(
    split: str = "val",
    max_rows: Optional[int] = None,
    batch_size: int = 50_000,
    max_history: Optional[int] = None,
) -> ImpressionList:
    """
    Load EB-NeRD behaviors for a given split into memory.
    Uses stream_ebnerd_behaviors() internally.
    """
    beh_key  = f"ebnerd_{split}_behaviors"
    hist_key = f"ebnerd_{split}_history"

    beh_path  = PATHS[beh_key]
    hist_path = PATHS.get(hist_key, Path("__nonexistent__"))

    log.info(f"Loading EB-NeRD [{split}] behaviors ...")
    hist_df   = _load_ebnerd_history_df(hist_path)
    hist_dict = _make_history_dict(hist_df, max_history=max_history)
    log.info(f"  History loaded for {len(hist_dict):,} users")

    behaviors: ImpressionList = []
    for batch in stream_ebnerd_behaviors(beh_path, hist_dict,
                                          batch_size=batch_size,
                                          max_rows=max_rows):
        behaviors.extend(batch)
        if max_rows and len(behaviors) >= max_rows:
            behaviors = behaviors[:max_rows]
            break

    log.info(f"  Loaded {len(behaviors):,} EB-NeRD impressions [{split}]")
    return behaviors


# ─────────────────────────────────────────────────────────────────────────────
# 5. MIND Behavior Loader
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mind_time(time_str: str) -> Optional[datetime]:
    """Parse MIND time format: '11/11/2019 9:05:58 AM'."""
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str.strip(), "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        try:
            return datetime.strptime(time_str.strip(), "%m/%d/%Y %H:%M:%S")
        except ValueError:
            return None


def stream_mind_behaviors(
    behaviors_path: Path,
    batch_size: int = 50_000,
    max_rows: Optional[int] = None,
) -> Iterator[ImpressionList]:
    """
    Memory-safe streaming generator for MIND behaviors.tsv.
    REUSES: ire-assignment1/data_utils.py::load_behaviors() logic.
    EXTENDS: streaming + rich impression schema.

    NOTE: ignore_errors=True is essential for MIND TSV (unescaped quotes).
    """
    beh_lazy = pl.scan_csv(
        behaviors_path,
        separator="\t",
        has_header=False,
        new_columns=MIND_BEHAVIOR_COLS,
        quote_char=None,
        ignore_errors=True,
    )
    total_rows = beh_lazy.select(pl.len()).collect().item()
    if max_rows:
        total_rows = min(total_rows, max_rows)

    log.info(f"  Streaming {total_rows:,} rows from {behaviors_path.name}")

    for offset in range(0, total_rows, batch_size):
        chunk = min(batch_size, total_rows - offset)
        batch_df = beh_lazy.slice(offset, chunk).collect()

        records: ImpressionList = []
        for row in batch_df.iter_rows(named=True):
            hist_raw = (row["history"] or "").strip()
            history  = hist_raw.split() if hist_raw else []

            imps_raw = (row["impressions"] or "").strip()
            raw_imps = imps_raw.split() if imps_raw else []

            candidates, labels = [], []
            is_labeled = bool(raw_imps) and "-" in raw_imps[0]

            for item in raw_imps:
                if is_labeled:
                    nid, lbl = item.rsplit("-", 1)
                    candidates.append(nid)
                    labels.append(int(lbl))
                else:
                    candidates.append(item)

            records.append({
                "impression_id"     : int(row["impression_id"]),
                "user_id"           : str(row["user_id"]),
                "impression_time"   : _parse_mind_time(row["time"]),
                "history"           : history,
                "history_times"     : [],   # not available in MIND
                "history_read_times": [],
                "history_scroll"    : [],
                "candidates"        : candidates,
                "labels"            : labels if is_labeled else None,
                "session_id"        : None,
                "read_time"         : None,
                "scroll_percentage" : None,
                "device_type"       : None,
                "is_subscriber"     : None,
                "age"               : None,
                "gender"            : None,
            })

        yield records


def load_mind_behaviors(
    split: str = "val",
    max_rows: Optional[int] = None,
    batch_size: int = 50_000,
) -> ImpressionList:
    """Load MIND behaviors for a given split into memory."""
    key = f"mind_{split}_behaviors"
    path = PATHS[key]

    log.info(f"Loading MIND [{split}] behaviors from {path} ...")
    behaviors: ImpressionList = []
    for batch in stream_mind_behaviors(path, batch_size=batch_size, max_rows=max_rows):
        behaviors.extend(batch)
        if max_rows and len(behaviors) >= max_rows:
            behaviors = behaviors[:max_rows]
            break

    log.info(f"  Loaded {len(behaviors):,} MIND impressions [{split}]")
    return behaviors


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public Router Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_behaviors(
    dataset: str,
    split: str = "val",
    max_rows: Optional[int] = None,
    sample_size: Optional[int] = None,
    batch_size: int = 50_000,
) -> ImpressionList:
    """
    Main entry point. Loads impressions for dataset+split.

    Args:
        dataset    : 'ebnerd' or 'mind'
        split      : 'train', 'val', or 'test'
        max_rows   : optional cap (useful for debug runs)
        sample_size: alias for max_rows
        batch_size : streaming batch size (default 50_000)

    Returns:
        List of unified impression dicts.
    """
    cap = sample_size if sample_size is not None else max_rows
    if dataset == "ebnerd":
        return load_ebnerd_behaviors(split=split, max_rows=cap,
                                      batch_size=batch_size)
    elif dataset == "mind":
        return load_mind_behaviors(split=split, max_rows=cap,
                                    batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'ebnerd' or 'mind'.")


def stream_behaviors(
    dataset: str,
    split: str = "test",
    batch_size: int = 50_000,
    max_rows: Optional[int] = None,
    max_history: Optional[int] = None,
) -> Iterator[ImpressionList]:
    """
    Streaming iterator for large test sets (13.5M EB-NeRD, 2.37M MIND).
    Use this instead of load_behaviors() for test-set prediction.

    max_history:
        Cap on retained clicks per user (see _make_history_dict). Required to
        keep the 13.5M-row EB-NeRD test set traversable in memory.
    """
    if dataset == "ebnerd":
        beh_path  = PATHS[f"ebnerd_{split}_behaviors"]
        hist_path = PATHS.get(f"ebnerd_{split}_history", Path("__nonexistent__"))
        hist_df   = _load_ebnerd_history_df(hist_path)
        hist_dict = _make_history_dict(hist_df, max_history=max_history)
        yield from stream_ebnerd_behaviors(beh_path, hist_dict,
                                            batch_size=batch_size,
                                            max_rows=max_rows)
    elif dataset == "mind":
        beh_path = PATHS[f"mind_{split}_behaviors"]
        yield from stream_mind_behaviors(beh_path, batch_size=batch_size,
                                          max_rows=max_rows)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Quick Sanity Print
# ─────────────────────────────────────────────────────────────────────────────

def _print_sample_summary(behaviors: ImpressionList, dataset: str, split: str):
    """Print a quick sanity summary of loaded behaviors."""
    total = len(behaviors)
    with_labels  = sum(1 for b in behaviors if b["labels"] is not None)
    has_history  = sum(1 for b in behaviors if b["history"])
    avg_cands    = sum(len(b["candidates"]) for b in behaviors) / max(1, total)
    avg_hist     = sum(len(b["history"]) for b in behaviors) / max(1, total)
    pos_rate     = 0.0
    if with_labels > 0:
        all_labels = [l for b in behaviors for l in (b["labels"] or [])]
        pos_rate   = sum(all_labels) / max(1, len(all_labels))

    log.info(f"\n{'─'*55}")
    log.info(f"  DATA SUMMARY: {dataset.upper()} [{split}]")
    log.info(f"{'─'*55}")
    log.info(f"  Total impressions : {total:,}")
    log.info(f"  With labels       : {with_labels:,}")
    log.info(f"  With history      : {has_history:,} ({has_history/max(1,total)*100:.1f}%)")
    log.info(f"  Avg candidates    : {avg_cands:.1f}")
    log.info(f"  Avg history len   : {avg_hist:.1f}")
    log.info(f"  Positive rate     : {pos_rate:.4f}")
    if behaviors:
        b0 = behaviors[0]
        log.info(f"  Sample imp_id     : {b0['impression_id']}")
        log.info(f"  Sample user_id    : {b0['user_id']}")
        log.info(f"  Sample candidates : {b0['candidates'][:4]}...")
        log.info(f"  Sample labels     : {(b0['labels'] or [])[:4]}...")
        log.info(f"  Sample history    : {b0['history'][:4]}...")
    log.info(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified data loader for EB-NeRD and MIND (Phase 1)"
    )
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Limit impressions loaded (debug mode)")
    parser.add_argument("--batch_size", type=int, default=50_000)
    args = parser.parse_args()

    from src.logger import log_section
    log_section(log, f"Phase 1: Data Loader — {args.dataset.upper()} [{args.split}]")

    # Load articles
    articles = load_articles(args.dataset, split=args.split)
    log.info(f"Articles loaded: {len(articles):,}")

    # Load behaviors
    behaviors = load_behaviors(
        args.dataset,
        split=args.split,
        max_rows=args.sample_size,
        batch_size=args.batch_size,
    )

    _print_sample_summary(behaviors, args.dataset, args.split)

    # Quick leakage check on EB-NeRD (has timestamps)
    if args.dataset == "ebnerd":
        violations = 0
        for b in behaviors[:1000]:
            imp_time = b.get("impression_time")
            if imp_time is None:
                continue
            for ht in b.get("history_times", []):
                if ht is not None and ht > imp_time:
                    violations += 1
        if violations:
            log.warning(f"  [LEAKAGE CHECK] {violations} future history timestamps found!")
        else:
            log.info("  [LEAKAGE CHECK] ✓ No future history timestamps detected (checked 1000 impressions)")


if __name__ == "__main__":
    main()
