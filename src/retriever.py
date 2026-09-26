"""
src/retriever.py — BM25, Dense, and Hybrid retrievers for news recommendation.

Assignment 2: News Recommendation System
IRE CS4.406

REUSES (from ire-assignment1):
  - BM25Retriever from bm25_retriever.py: inverted index, IDF, candidate scoring
  - evaluate_recall logic from candidate_retriever.py

EXTENDS WITH:
  - DenseRetriever: TF-IDF + TruncatedSVD(128) + PyTorch cosine dot product (device-aware)
  - HybridRetriever: Reciprocal Rank Fusion (RRF) combining BM25 and Dense
  - retrieve_topk() for global catalog candidate retrieval
  - score_candidates() / rank_impression() for in-impression candidate scoring
  - print_recall_sanity() check with configurable warning threshold

CLI USAGE:
  .venv/bin/python3 src/retriever.py --dataset mind --method bm25 --sample_size 200
  .venv/bin/python3 src/retriever.py --dataset ebnerd --method dense --sample_size 200
  .venv/bin/python3 src/retriever.py --dataset mind --method hybrid --sample_size 200
"""

import argparse
import hashlib
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.config import DEVICE, HYPERPARAMS, PATHS
from src.data_loader import ArticleDict, ImpressionList, load_articles, load_behaviors
from src.logger import get_logger

log = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """Lightweight alphanumeric tokenizer matching A1."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# 1. BM25 Retriever
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Inverted-index BM25 candidate ranker and global retriever.
    Reuses and extends A1 BM25Retriever.
    """
    def __init__(self, articles: ArticleDict, k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.article_ids = list(articles.keys())
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.doc_freqs: Counter = Counter()
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.avgdl = 1.0
        self.N = len(articles)
        self.idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        log.info(f"Building BM25 inverted index for {self.N:,} articles ...")
        total_tokens = 0
        for aid, meta in self.articles.items():
            tokens = tokenize(meta.get("text", ""))
            self.doc_len[aid] = len(tokens)
            total_tokens += len(tokens)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][aid] = count

        self.avgdl = (total_tokens / self.N) if self.N > 0 else 1.0
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self.doc_freqs.items()
        }
        log.info(f"BM25 index built: {len(self.inverted_index):,} terms, avgdl={self.avgdl:.1f}")

    def build_user_query(self, history: List[str], max_history: int = 10) -> List[str]:
        """Constructs query tokens from titles of recently clicked articles."""
        if not history:
            return []
        recent_history = history[-max_history:]
        query_text = " ".join([self.articles.get(aid, {}).get("title", "") for aid in recent_history])
        return tokenize(query_text)

    def score_candidates(self, query_tokens: List[str], candidate_ids: List[str]) -> List[float]:
        """Scores given candidate articles against user query tokens."""
        if not query_tokens or not candidate_ids:
            return [0.0] * len(candidate_ids)

        q_counter = Counter(query_tokens)
        scores: List[float] = []
        for aid in candidate_ids:
            score = 0.0
            dl = self.doc_len.get(aid, self.avgdl)
            for term, _ in q_counter.items():
                if term in self.inverted_index and aid in self.inverted_index[term]:
                    tf = self.inverted_index[term][aid]
                    idf = self.idf.get(term, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    score += idf * (numerator / denominator)
            scores.append(score)
        return scores

    def rank_impression(self, history: List[str], candidate_ids: List[str], max_history: int = 10) -> List[float]:
        """Scores candidate articles for a single impression."""
        query_tokens = self.build_user_query(history, max_history=max_history)
        return self.score_candidates(query_tokens, candidate_ids)

    def retrieve_topk(self, history: List[str], k: int = 200, max_history: int = 10) -> List[Tuple[str, float]]:
        """Global catalog retrieval of top-K articles matching user history."""
        query_tokens = self.build_user_query(history, max_history=max_history)
        if not query_tokens:
            return [(aid, 0.0) for aid in self.article_ids[:k]]

        q_counter = Counter(query_tokens)
        scores: Dict[str, float] = {}
        for term, _ in q_counter.items():
            if term in self.inverted_index:
                idf = self.idf.get(term, 0.0)
                if idf <= 0:
                    continue
                for aid, tf in self.inverted_index[term].items():
                    dl = self.doc_len.get(aid, self.avgdl)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    scores[aid] = scores.get(aid, 0.0) + idf * (tf * (self.k1 + 1) / denom)

        if not scores:
            return [(aid, 0.0) for aid in self.article_ids[:k]]

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        if len(sorted_items) < k:
            seen = {aid for aid, _ in sorted_items}
            for aid in self.article_ids:
                if aid not in seen:
                    sorted_items.append((aid, 0.0))
                    if len(sorted_items) >= k:
                        break
        return sorted_items


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dense Semantic Retriever (TF-IDF + TruncatedSVD + PyTorch Dot Product)
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Dense semantic retriever with a selectable backend.

    backend="minilm" (default)
        SentenceTransformer `all-MiniLM-L6-v2` with L2-normalised embeddings,
        matching ire-assignment1/dense_retriever.py exactly: same model, same
        `normalize_embeddings=True`, same mean-pooled+L2-normalised user vector,
        same inner-product scoring. Optional FAISS IndexFlatIP for global top-K.
        This is the A1-parity backend and the one to use for reported results.

    backend="tfidf_svd"
        The original cheap path: TF-IDF + TruncatedSVD. No pretrained semantics,
        no model download, ~1s to fit. Retained as (a) an offline fallback and
        (b) a legitimate ablation arm -- on MIND it scores ~0.08 AUC below MiniLM,
        so the two must never be silently mixed.

    Scoring semantics are identical across backends (cosine similarity between a
    mean-pooled user vector and each candidate), so the `dense_score` feature means
    the same thing either way -- but its VALUE distribution differs, which is why
    LightGBMReranker fingerprints the backend and refuses a mismatched checkpoint.
    """
    def __init__(
        self,
        articles: ArticleDict,
        backend: str = "minilm",
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 128,
        max_features: int = 40000,
        device: Optional[str] = None,
        batch_size: int = 256,
        use_faiss: bool = True,
    ):
        self.articles = articles
        self.article_ids = list(articles.keys())
        self.aid2idx = {aid: i for i, aid in enumerate(self.article_ids)}
        self.backend = backend
        self.model_name = model_name
        self.max_history_standalone = 20

        if backend == "minilm":
            embeddings_np = self._fit_minilm(device=device, batch_size=batch_size)
        elif backend == "tfidf_svd":
            embeddings_np = self._fit_tfidf_svd(dim=dim, max_features=max_features)
        else:
            raise ValueError(f"Unknown dense backend: {backend!r} (use 'minilm' or 'tfidf_svd')")

        # Numpy mirror: the hot prediction path. Scoring a ~50-candidate slate
        # against a 384-d query is microseconds of BLAS, whereas four separate
        # torch/MPS kernel launches per impression dominate the profile at
        # 13.5M-row test-set scale. Verified numerically identical to the torch
        # path (max abs diff 1.8e-07) and responsible for a 5.4x speedup.
        self._emb_np = np.ascontiguousarray(embeddings_np, dtype=np.float32)
        self.dim = self._emb_np.shape[1]

        # Torch view, kept for the reference/GPU path and for HybridRetriever.
        self.device = device or DEVICE
        self.article_embeddings = torch.tensor(
            self._emb_np, dtype=torch.float32, device=self.device
        )

        # Optional FAISS index, mirroring A1's global-retrieval backend.
        self.faiss_index = None
        if use_faiss and backend == "minilm":
            try:
                import faiss

                index = faiss.IndexFlatIP(self.dim)
                index.add(self._emb_np)
                self.faiss_index = index
                log.info(f"FAISS IndexFlatIP built (dim={self.dim})")
            except Exception as e:
                log.info(f"FAISS unavailable ({type(e).__name__}); using numpy top-K instead.")

        log.info(
            f"Dense retriever ready: backend={backend} dim={self.dim} "
            f"articles={len(self.article_ids):,}"
        )

    # ── Backends ──────────────────────────────────────────────────────────────

    def _catalog_fingerprint(self) -> str:
        """
        Cheap, stable identity for (catalog contents, model).

        Uses a hash of the article ID list plus the model name rather than the
        article texts, so it costs no extra I/O. Text changes with the same IDs
        would not be detected, which is acceptable here because the only text
        inputs are the shipped article files.
        """
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(str(len(self.article_ids)).encode())
        for aid in self.article_ids:
            h.update(aid.encode())
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def _cache_path(self) -> Optional[Path]:
        """
        Embeddings cache location, or None when caching is disabled.

        NOTE: a bare `except Exception: return None` here previously swallowed a
        NameError (PATHS was not imported in this module), which silently
        disabled caching with no log line. The error is now surfaced.
        """
        if not HYPERPARAMS.get("dense_cache", True):
            return None
        try:
            d = Path(PATHS["features"]).parent / "dense_cache"
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{self._catalog_fingerprint()}.npy"
        except Exception as e:
            log.warning(f"[dense] Embedding cache unavailable ({type(e).__name__}: {e})")
            return None

    def _encode_device(self) -> str:

        return str(HYPERPARAMS.get("dense_encode_device", "cpu"))

    def _fit_minilm(self, device: Optional[str], batch_size: int) -> np.ndarray:

        import torch as _torch
        from sentence_transformers import SentenceTransformer

        cache = self._cache_path()
        if cache is not None and cache.exists():
            try:
                emb = np.load(cache)
                if emb.shape[0] == len(self.article_ids):
                    log.info(
                        f"MiniLM embeddings loaded from cache {cache.name} "
                        f"-> shape {emb.shape}"
                    )
                    return emb
                log.info(f"Cache {cache.name} shape mismatch; re-encoding.")
            except Exception as e:
                log.info(f"Cache read failed ({type(e).__name__}); re-encoding.")

        dev = device or self._encode_device()
        # config.py pins OMP_NUM_THREADS=1 process-wide to avoid a torch/MPS
        # deadlock, which leaves the encode single-threaded. Raise it for the
        # encode only; scoring uses numpy and is unaffected.
        prev_threads = _torch.get_num_threads()
        want_threads = int(HYPERPARAMS.get("dense_torch_threads", 4))
        if want_threads > prev_threads:
            _torch.set_num_threads(want_threads)

        log.info(
            f"Encoding {len(self.article_ids):,} articles with "
            f"SentenceTransformer({self.model_name}) on {dev} "
            f"(batch={batch_size}, torch_threads={_torch.get_num_threads()}) ..."
        )
        t0 = time.perf_counter()
        try:
            model = SentenceTransformer(self.model_name, device=dev)
            texts = [self.articles[aid].get("text", "") for aid in self.article_ids]
            emb = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,   # matches A1: dot product == cosine
                convert_to_numpy=True,
            ).astype(np.float32)
        finally:
            _torch.set_num_threads(prev_threads)
        log.info(
            f"MiniLM encoding finished in {time.perf_counter() - t0:.1f}s "
            f"-> shape {emb.shape}"
        )

        if cache is not None:
            try:
                np.save(cache, emb)
                log.info(f"Cached embeddings to {cache}")
            except Exception as e:
                log.info(f"Cache write failed ({type(e).__name__}); continuing.")
        return emb

    def _fit_tfidf_svd(self, dim: int, max_features: int) -> np.ndarray:
        """Original cheap backend: TF-IDF + TruncatedSVD, L2-normalised."""
        log.info(f"Fitting TF-IDF + TruncatedSVD({dim}) for {len(self.article_ids):,} articles ...")
        texts = [self.articles[aid].get("text", "") for aid in self.article_ids]
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        tfidf_mat = self.vectorizer.fit_transform(texts)
        actual_dim = min(dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        emb = self.svd.fit_transform(tfidf_mat).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return emb / norms

    # ── Numpy hot path ────────────────────────────────────────────────────────

    def compute_user_embedding_np(self, history: List[str], max_history: int = 20) -> np.ndarray:
        """
        Numpy equivalent of compute_user_embedding(), used on the hot
        prediction path. Returns a (dim,) float32 array.
        """
        if not history:
            return np.zeros(self.dim, dtype=np.float32)

        recent = history[-max_history:]
        idx = [self.aid2idx[a] for a in recent if a in self.aid2idx]
        if not idx:
            return np.zeros(self.dim, dtype=np.float32)

        user_emb = self._emb_np[idx].mean(axis=0)
        n = np.linalg.norm(user_emb)
        if n > 0:
            user_emb = user_emb / n
        return user_emb.astype(np.float32)

    def score_candidates_np(self, user_emb: np.ndarray, candidate_ids: List[str]) -> List[float]:
        """Numpy equivalent of score_candidates()."""
        if not candidate_ids or not np.any(user_emb):
            return [0.0] * len(candidate_ids)

        idx = [self.aid2idx.get(a, -1) for a in candidate_ids]
        if all(i >= 0 for i in idx):
            return (self._emb_np[idx] @ user_emb).tolist()
        return [
            float(self._emb_np[i] @ user_emb) if i >= 0 else 0.0
            for i in idx
        ]

    def rank_impression_np(self, history: List[str], candidate_ids: List[str], max_history: int = 20) -> List[float]:
        """Numpy hot path for ranking an impression's candidates."""
        return self.score_candidates_np(self.compute_user_embedding_np(history, max_history), candidate_ids)

    # ── Torch reference path ──────────────────────────────────────────────────

    def compute_user_embedding(self, history: List[str], max_history: int = 20) -> torch.Tensor:
        """Computes mean-pooled, normalized user embedding on device."""
        if not history:
            return torch.zeros(self.dim, device=self.device, dtype=torch.float32)

        recent_history = history[-max_history:]
        valid_indices = [self.aid2idx[aid] for aid in recent_history if aid in self.aid2idx]

        if not valid_indices:
            return torch.zeros(self.dim, device=self.device, dtype=torch.float32)

        idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
        user_emb = torch.mean(self.article_embeddings[idx_tensor], dim=0)
        norm = torch.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm
        return user_emb

    def score_candidates(self, user_emb: torch.Tensor, candidate_ids: List[str]) -> List[float]:
        """Scores candidate articles using device dot product."""
        if torch.all(user_emb == 0) or not candidate_ids:
            return [0.0] * len(candidate_ids)

        cand_indices = [self.aid2idx.get(aid, -1) for aid in candidate_ids]
        scores: List[float] = []

        # Fast path: check valid indices
        valid_cands = [i for i in cand_indices if i != -1]
        if len(valid_cands) == len(candidate_ids):
            c_tensor = torch.tensor(cand_indices, dtype=torch.long, device=self.device)
            c_embs = self.article_embeddings[c_tensor]
            sims = torch.matmul(c_embs, user_emb).cpu().tolist()
            return sims

        for idx in cand_indices:
            if idx != -1:
                score = torch.dot(user_emb, self.article_embeddings[idx]).item()
            else:
                score = 0.0
            scores.append(score)
        return scores

    def rank_impression(self, history: List[str], candidate_ids: List[str], max_history: int = 20) -> List[float]:
        """Ranks candidate articles for a single impression."""
        user_emb = self.compute_user_embedding(history, max_history=max_history)
        return self.score_candidates(user_emb, candidate_ids)

    # ── Global retrieval ──────────────────────────────────────────────────────

    def retrieve_topk(self, history: List[str], k: int = 200, max_history: int = 20) -> List[Tuple[str, float]]:
        """
        Global catalog retrieval. Uses FAISS IndexFlatIP when available (A1's
        backend), otherwise an exact numpy/torch top-K over the full matrix.
        Both are exact inner-product search, so they agree up to float error.
        """
        if self.faiss_index is not None:
            user_emb = self.compute_user_embedding_np(history, max_history=max_history)
            if not np.any(user_emb):
                return [(aid, 0.0) for aid in self.article_ids[:k]]
            kk = min(k, len(self.article_ids))
            scores, indices = self.faiss_index.search(
                np.expand_dims(user_emb.astype(np.float32), axis=0), kk
            )
            return [
                (self.article_ids[int(idx)], float(s))
                for idx, s in zip(indices[0], scores[0])
                if idx != -1
            ]

        user_emb = self.compute_user_embedding(history, max_history=max_history)
        if torch.all(user_emb == 0):
            return [(aid, 0.0) for aid in self.article_ids[:k]]

        sims = torch.matmul(self.article_embeddings, user_emb)
        topk_vals, topk_inds = torch.topk(sims, min(k, len(self.article_ids)))
        return [
            (self.article_ids[idx], float(val))
            for idx, val in zip(topk_inds.cpu().numpy(), topk_vals.cpu().numpy())
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Retriever (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever combining BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF).
    score(d) = 1 / (c + rank_bm25(d)) + 1 / (c + rank_dense(d)), where c=60.
    """
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_c: float = 60.0):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_c = rrf_c
        self.article_ids = bm25.article_ids

    def score_candidates(
        self,
        history: List[str],
        candidate_ids: List[str],
        max_history: int = 10,
    ) -> List[float]:
        """Computes RRF score for candidates within an impression."""
        if not candidate_ids:
            return []

        bm25_scores = self.bm25.rank_impression(history, candidate_ids, max_history=max_history)
        dense_scores = self.dense.rank_impression(history, candidate_ids, max_history=max_history)

        # Convert scores to rank orders (0-indexed rank: highest score -> rank 0)
        bm25_order = np.argsort(-np.array(bm25_scores))
        bm25_ranks = np.empty_like(bm25_order)
        bm25_ranks[bm25_order] = np.arange(len(bm25_scores))

        dense_order = np.argsort(-np.array(dense_scores))
        dense_ranks = np.empty_like(dense_order)
        dense_ranks[dense_order] = np.arange(len(dense_scores))

        rrf_scores = []
        for i in range(len(candidate_ids)):
            s = (1.0 / (self.rrf_c + bm25_ranks[i])) + (1.0 / (self.rrf_c + dense_ranks[i]))
            rrf_scores.append(float(s))
        return rrf_scores

    def rank_impression(self, history: List[str], candidate_ids: List[str], max_history: int = 10) -> List[float]:
        return self.score_candidates(history, candidate_ids, max_history=max_history)

    def retrieve_topk(self, history: List[str], k: int = 200, fetch_k: int = 500) -> List[Tuple[str, float]]:
        """Retrieves top-K global candidates by fusing top-fetch_k from BM25 and Dense."""
        bm25_top = self.bm25.retrieve_topk(history, k=fetch_k)
        dense_top = self.dense.retrieve_topk(history, k=fetch_k)

        rrf_dict: Dict[str, float] = {}
        for rank, (aid, _) in enumerate(bm25_top):
            rrf_dict[aid] = rrf_dict.get(aid, 0.0) + (1.0 / (self.rrf_c + rank))

        for rank, (aid, _) in enumerate(dense_top):
            rrf_dict[aid] = rrf_dict.get(aid, 0.0) + (1.0 / (self.rrf_c + rank))

        sorted_items = sorted(rrf_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_items


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Factory: build a DenseRetriever from HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────────────

def build_dense_retriever(
    articles: ArticleDict,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    **overrides,
) -> "DenseRetriever":
    """
    Builds a DenseRetriever using the configured backend.

    Centralising this matters: `dense_score` is LightGBM's single most important
    feature on MIND (17.8% of total gain, measured), and `hybrid_score` adds
    another 14.0%, so ~32% of the MIND model's signal flows through this object.
    If different call sites silently construct different backends, the re-ranker
    is trained on one semantic space and served in another. Every call site
    therefore goes through here and reads the same HYPERPARAMS keys.
    """
    kwargs = {
        "backend": backend or HYPERPARAMS.get("dense_backend", "minilm"),
        "model_name": HYPERPARAMS.get("dense_model_name", "all-MiniLM-L6-v2"),
        "dim": HYPERPARAMS.get("svd_n_components", 128),
        "max_features": HYPERPARAMS.get("tfidf_max_features", 50_000),
        "batch_size": HYPERPARAMS.get("dense_batch_size", 256),
        "use_faiss": HYPERPARAMS.get("dense_use_faiss", True),
    }
    kwargs.update(overrides)
    if device is not None:
        kwargs["device"] = device
    return DenseRetriever(articles, **kwargs)


def catalog_fingerprint(article_ids: List[str]) -> str:
    """
    Stable identity of the catalogue the retrieval index was built over.

    Same hash as DenseRetriever._catalog_fingerprint, exposed for the checkpoint
    fingerprint. It must be part of that: MIND's per-split news.tsv files share
    only 28,460 of ~50K IDs, so training on one index and serving another shifts
    bm25_score (mean 6.98 -> 8.53) and dense_score (0.115 -> 0.140) without
    changing the column count at all.
    """
    h = hashlib.sha256()
    h.update(str(len(article_ids)).encode())
    for aid in article_ids:
        h.update(str(aid).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def feature_fingerprint(
    article_ids: Optional[List[str]] = None,
    train_config: Optional[dict] = None,
) -> Dict[str, str]:

    fp = {
        "dense_backend": HYPERPARAMS.get("dense_backend", "minilm"),
        "dense_model_name": HYPERPARAMS.get("dense_model_name", "all-MiniLM-L6-v2"),
    }
    if article_ids is not None:
        fp["index_catalog"] = f"{len(article_ids)}:{catalog_fingerprint(article_ids)}"
    if train_config:
        for k, v in train_config.items():
            fp[f"train_{k}"] = str(v)
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation & Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], ground_truth_clicked: List[str], k: int) -> float:
    """Computes Recall@K: proportion of ground-truth clicks in top-K retrieved."""
    if not ground_truth_clicked:
        return 0.0
    topk_set = set(retrieved_ids[:k])
    hits = sum(1 for aid in ground_truth_clicked if aid in topk_set)
    return hits / len(ground_truth_clicked)


def evaluate_recall(
    retriever,
    behaviors: ImpressionList,
    k_values: List[int] = [50, 100, 200],
    max_eval: int = 500,
) -> Dict[str, float]:
    """Computes average Recall@K over a set of impression behaviors."""
    recall_sums = {f"Recall@{k}": 0.0 for k in k_values}
    total_valid = 0
    max_k = max(k_values)

    eval_behaviors = behaviors[:max_eval] if max_eval else behaviors

    for imp in tqdm(eval_behaviors, desc="Evaluating Recall", leave=False):
        labels = imp.get("labels")
        if labels is None:
            continue

        candidates = imp.get("candidates", [])
        clicked = [c for c, l in zip(candidates, labels) if l == 1]
        if not clicked:
            continue

        history = imp.get("history", [])
        topk_tuples = retriever.retrieve_topk(history, k=max_k)
        retrieved_ids = [aid for aid, _ in topk_tuples]

        for k in k_values:
            recall_sums[f"Recall@{k}"] += recall_at_k(retrieved_ids, clicked, k=k)
        total_valid += 1

    if total_valid == 0:
        return {f"Recall@{k}": 0.0 for k in k_values}

    return {k: v / total_valid for k, v in recall_sums.items()}


def print_recall_sanity(
    retriever,
    behaviors: ImpressionList,
    method_name: str,
    k_values: List[int] = [50, 100, 200],
    warn_threshold: float = 0.15,
) -> Dict[str, float]:
    """Prints recall sanity stats and emits a warning if Recall@200 < warn_threshold."""
    metrics = evaluate_recall(retriever, behaviors, k_values=k_values)
    metric_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    log.info(f"[RECALL SANITY] method={method_name:<7} | {metric_str}")

    r200 = metrics.get("Recall@200", 0.0)
    if r200 < warn_threshold:
        log.warning(
            f"WARN: Recall@200 ({r200:.4f}) < {warn_threshold:.2f} — retrieval quality is low!"
        )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Retriever module (BM25, Dense, Hybrid)")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="mind", help="Dataset to test on")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25", help="Retrieval method")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of validation impressions to evaluate")
    parser.add_argument("--split", default="val", help="Data split to evaluate on")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"  Phase 2: Retriever — {args.dataset.upper()} [{args.method}]")
    log.info("=" * 60)

    # 1. Load articles & behaviors
    articles = load_articles(args.dataset, split=args.split)
    behaviors = load_behaviors(args.dataset, split=args.split, sample_size=args.sample_size)

    # 2. Build requested retriever
    if args.method == "bm25":
        retriever = BM25Retriever(
            articles,
            k1=HYPERPARAMS.get("bm25_k1", 1.5),
            b=HYPERPARAMS.get("bm25_b", 0.75),
        )
    elif args.method == "dense":
        retriever = DenseRetriever(
            articles,
            dim=HYPERPARAMS.get("svd_n_components", 128),
            max_features=HYPERPARAMS.get("tfidf_max_features", 40000),
        )
    elif args.method == "hybrid":
        bm25 = BM25Retriever(
            articles,
            k1=HYPERPARAMS.get("bm25_k1", 1.5),
            b=HYPERPARAMS.get("bm25_b", 0.75),
        )
        dense = DenseRetriever(
            articles,
            dim=HYPERPARAMS.get("svd_n_components", 128),
            max_features=HYPERPARAMS.get("tfidf_max_features", 40000),
        )
        retriever = HybridRetriever(bm25, dense, rrf_c=HYPERPARAMS.get("rrf_k", 60.0))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 3. Sanity check recall
    metrics = print_recall_sanity(retriever, behaviors, method_name=args.method)

    # 4. Demonstrate ranking a sample impression
    sample_imp = next((b for b in behaviors if b.get("labels")), behaviors[0])
    scores = retriever.rank_impression(sample_imp.get("history", []), sample_imp["candidates"])
    log.info(f"Sample impression {sample_imp['impression_id']} ranking scores: {[round(s, 4) for s in scores[:5]]} ...")
    log.info("Phase 2 sanity check complete.")


if __name__ == "__main__":
    main()
