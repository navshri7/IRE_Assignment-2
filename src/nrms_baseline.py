"""
src/nrms_baseline.py -- Pure-PyTorch reproduction of the NRMS baseline.

Reproduces the architecture from:
  ebnerd-benchmark/examples/quick_start/nrms_ebnerd.py  (EB-NeRD)
with a parallel MIND adapter.

Architecture matches ebrec/models/newsrec/nrms.py exactly:
  - NewsEncoder: Token Embedding -> Dropout -> Multi-Head Self-Attention -> Additive Attention Pool
  - UserEncoder: TimeDistributed(NewsEncoder) -> Multi-Head Self-Attention -> Additive Attention Pool
  - Scoring: dot(user_vec, news_vec) -- softmax during training, sigmoid for scoring

Hyperparameters mirror hparams_nrms from model_config.py:
  title_size=30, history_size=20, head_num=20, head_dim=20,
  attention_hidden_dim=200, dropout=0.2, lr=1e-4

Training strategy mirrors Wu et al. 2019 (negative sampling, npratio=4):
  1 positive + 4 random negatives per impression (like sampling_strategy_wu2019)

CLI:
  .venv/bin/python3 src/nrms_baseline.py --dataset ebnerd --epochs 5 --sample_size 2000
  .venv/bin/python3 src/nrms_baseline.py --dataset mind   --epochs 5 --sample_size 2000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import DEVICE, PATHS
from src.data_loader import ArticleDict, ImpressionList, load_articles, load_behaviors
from src.evaluator import evaluate_all_slices, format_results_markdown
from src.logger import get_logger

log = get_logger(__name__)

# Official hparams_nrms (from ebrec/models/newsrec/model_config.py)
TITLE_SIZE = 30           # max tokens per article title
HISTORY_SIZE = 20         # max clicked articles in user history
HEAD_NUM = 20             # number of attention heads
HEAD_DIM = 20             # dimension per head (output_dim = HEAD_NUM * HEAD_DIM = 400)
ATTENTION_HIDDEN_DIM = 200
DROPOUT = 0.2
LR = 1e-4
NPRATIO = 4               # negatives per positive (Wu 2019 section 3.2)
WORD_EMB_DIM = 100        # word embedding dimension

# ebrec: TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL, DEFAULT_BODY_COL]
NRMS_TEXT_COLUMNS = ("title", "abstract", "text_body")

# ebrec: TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_TRANSFORMER_MODEL = "FacebookAI/xlm-roberta-base"

# Word-embedding dim of the two models ebrec uses in its scripts.
XLM_R_DIM = 768          # xlm-roberta-base
BERT_DIM = 768           # bert-base-uncased


def article_text(art: dict) -> str:
    """
    Article text as ebrec builds it: pl.concat_str(title, subtitle, body,
    separator=" ").

    Our unified article dict exposes EB-NeRD's subtitle as `abstract` and the
    full body as `text_body` (populated only when available). MIND has no body,
    so its text is "title abstract" -- the closest available equivalent.
    """
    parts = []
    for col in NRMS_TEXT_COLUMNS:
        v = art.get(col)
        if v:
            parts.append(str(v))
    return " ".join(parts)


class NRMSVocab:
    """
    Token-ID encoder. Backends:

      "transformer" (default, faithful) -- wraps a HuggingFace tokenizer and
          reproduces convert_text2encoding_with_transformers exactly, including
          add_special_tokens=False and padding to a fixed max_length.
      "wordfreq" (legacy) -- top-N whitespace frequency vocabulary.
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(
        self,
        max_vocab: int = 30_000,
        title_size: int = TITLE_SIZE,
        backend: str = "transformer",
        model_name: str = DEFAULT_TRANSFORMER_MODEL,
    ):
        self.max_vocab = max_vocab
        self.title_size = title_size
        self.backend = backend
        self.model_name = model_name
        self.word2id: Dict[str, int] = {}
        self.tokenizer = None
        self.vocab_size = 2
        if backend == "transformer":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = len(self.tokenizer)
            log.info(
                f"NRMSVocab[transformer]: {model_name} vocab={self.vocab_size:,} "
                f"title_size={title_size}"
            )
        else:
            log.info(f"NRMSVocab[wordfreq] (LEGACY, not the official baseline)")

    def build(self, texts: List[str]) -> "NRMSVocab":
        if self.backend != "wordfreq":
            return self  # the pretrained tokenizer is already fixed
        from collections import Counter

        counts: Counter = Counter()
        for text in texts:
            for tok in (text or "").lower().split():
                counts[tok] += 1
        top_words = [w for w, _ in counts.most_common(self.max_vocab - 2)]
        self.word2id = {w: i + 2 for i, w in enumerate(top_words)}
        self.vocab_size = len(self.word2id) + 2
        log.info(f"NRMSVocab[wordfreq]: {self.vocab_size:,} words from {len(texts):,} articles")
        return self

    def encode(self, text: str) -> List[int]:
        if self.backend == "transformer":
            # Mirrors convert_text2encoding_with_transformers: no special
            # tokens, padded to a fixed length, truncated.
            out = self.tokenizer(
                [text or ""],
                add_special_tokens=False,
                padding="max_length",
                max_length=self.title_size,
                truncation=True,
            )["input_ids"]
            return list(out[0])
        tokens = (text or "").lower().split()[:self.title_size]
        ids = [self.word2id.get(t, self.UNK_ID) for t in tokens]
        ids += [self.PAD_ID] * (self.title_size - len(ids))
        return ids[:self.title_size]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if self.backend == "transformer":
            out = self.tokenizer(
                list(texts),
                add_special_tokens=False,
                padding="max_length",
                max_length=self.title_size,
                truncation=True,
            )["input_ids"]
            return np.array(out, dtype=np.int64)
        return np.array([self.encode(t) for t in texts], dtype=np.int64)


def load_transformer_word_embeddings(model_name: str) -> np.ndarray:
    """
    The pretrained word-embedding matrix, matching ebrec's
    get_transformers_word_embeddings(model) = model.embeddings.word_embeddings
    .weight.data.cpu().numpy().
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    W = model.embeddings.word_embeddings.weight.data.to("cpu").numpy()
    log.info(f"Loaded pretrained word embeddings {model_name}: {W.shape}")
    return np.ascontiguousarray(W, dtype=np.float32)


def build_article_title_matrix(
    articles: ArticleDict,
    vocab: NRMSVocab,
) -> Tuple[Dict[str, int], np.ndarray]:
    """Build article_id->index mapping and token matrix. Index 0 = PAD row."""
    article_ids = list(articles.keys())
    texts = [article_text(articles[aid]) for aid in article_ids]
    encoded = vocab.encode_batch(texts)
    pad_row = np.zeros((1, vocab.title_size), dtype=np.int64)
    matrix = np.concatenate([pad_row, encoded], axis=0)
    article_id_to_idx: Dict[str, int] = {aid: i + 1 for i, aid in enumerate(article_ids)}
    return article_id_to_idx, matrix


# =============================================================================
# 2. PyTorch Layers
#    SelfAttention mirrors SelfAttention in ebrec/models/newsrec/layers.py
#    AdditiveAttention mirrors AttLayer2 in ebrec/models/newsrec/layers.py
# =============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention. output_dim = head_num * head_dim = 400."""

    def __init__(self, head_num: int = HEAD_NUM, head_dim: int = HEAD_DIM, input_dim: int = WORD_EMB_DIM):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.output_dim = head_num * head_dim
        self.scale = math.sqrt(head_dim)
        self.WQ = nn.Linear(input_dim, self.output_dim, bias=False)
        self.WK = nn.Linear(input_dim, self.output_dim, bias=False)
        self.WV = nn.Linear(input_dim, self.output_dim, bias=False)
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, L, _ = Q.shape
        q = self.WQ(Q).view(B, L, self.head_num, self.head_dim).transpose(1, 2)
        k = self.WK(K).view(B, L, self.head_num, self.head_dim).transpose(1, 2)
        v = self.WV(V).view(B, L, self.head_num, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(B, L, self.output_dim)


class AdditiveAttention(nn.Module):
    """Additive attention pooling (AttLayer2 in ebrec)."""

    def __init__(self, input_dim: int, hidden_dim: int = ATTENTION_HIDDEN_DIM):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.W(x))
        scores = self.q(h).squeeze(-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# =============================================================================
# 3. NRMS News Encoder
#    Mirrors _build_newsencoder() in ebrec/models/newsrec/nrms.py
# =============================================================================

class NewsEncoder(nn.Module):
    """Encodes article token IDs -> dense vector."""

    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int = WORD_EMB_DIM,
        head_num: int = HEAD_NUM,
        head_dim: int = HEAD_DIM,
        attention_hidden_dim: int = ATTENTION_HIDDEN_DIM,
        dropout: float = DROPOUT,
        pretrained_emb: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.output_dim = head_num * head_dim  # 400
        self.embedding = nn.Embedding(vocab_size, word_emb_dim)
        if pretrained_emb is not None:
            # ebrec: tf.keras.layers.Embedding(shape, weights=[word2vec],
            # trainable=True) -- initialised from the pretrained matrix, then
            # fine-tuned. Note no padding_idx: ebrec's pads carry a learned
            # vector, so none is set here either.
            if pretrained_emb.shape[0] < vocab_size or pretrained_emb.shape[1] != word_emb_dim:
                raise ValueError(
                    f"pretrained_emb shape {pretrained_emb.shape} incompatible with "
                    f"Embedding({vocab_size}, {word_emb_dim})"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(
                    torch.tensor(pretrained_emb[:vocab_size], dtype=torch.float32)
                )
            log.info(
                f"NewsEncoder: embedding initialised from pretrained matrix "
                f"{pretrained_emb.shape} (trainable, as in ebrec)"
            )
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = SelfAttention(head_num, head_dim, input_dim=word_emb_dim)
        self.drop2 = nn.Dropout(dropout)
        self.pool = AdditiveAttention(self.output_dim, attention_hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args: token_ids (B, title_size). Returns: (B, output_dim=400)."""
        x = self.embedding(token_ids)
        x = self.drop1(x)
        x = self.self_attn(x, x, x)
        x = self.drop2(x)
        return self.pool(x)


# =============================================================================
# 4. NRMS User Encoder
#    Mirrors _build_userencoder() in ebrec/models/newsrec/nrms.py
# =============================================================================

class UserEncoder(nn.Module):
    """Encodes user click history -> dense user vector."""

    def __init__(
        self,
        news_encoder: NewsEncoder,
        head_num: int = HEAD_NUM,
        head_dim: int = HEAD_DIM,
        attention_hidden_dim: int = ATTENTION_HIDDEN_DIM,
    ):
        super().__init__()
        self.news_encoder = news_encoder
        news_dim = news_encoder.output_dim
        self.self_attn = SelfAttention(head_num, head_dim, input_dim=news_dim)
        self.pool = AdditiveAttention(news_dim, attention_hidden_dim)

    def forward(self, history_tokens: torch.Tensor) -> torch.Tensor:
        """Args: history_tokens (B, H, T). Returns: (B, news_dim=400)."""
        B, H, T = history_tokens.shape
        flat = history_tokens.view(B * H, T)
        news_vecs = self.news_encoder(flat).view(B, H, -1)
        ctx = self.self_attn(news_vecs, news_vecs, news_vecs)
        return self.pool(ctx)


# =============================================================================
# 5. NRMS Full Model
#    Mirrors NRMSModel._build_nrms() in ebrec/models/newsrec/nrms.py
# =============================================================================

class NRMSModel(nn.Module):
    """
    NRMS (Wu et al. EMNLP 2019).
    Training mode:  (his_tokens, cand_tokens) -> CrossEntropy logits
    Inference mode: score_single() -> sigmoid score per candidate
    """

    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int = WORD_EMB_DIM,
        head_num: int = HEAD_NUM,
        head_dim: int = HEAD_DIM,
        attention_hidden_dim: int = ATTENTION_HIDDEN_DIM,
        dropout: float = DROPOUT,
        pretrained_emb: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(
            vocab_size, word_emb_dim, head_num, head_dim, attention_hidden_dim,
            dropout, pretrained_emb=pretrained_emb,
        )
        self.user_encoder = UserEncoder(self.news_encoder, head_num, head_dim, attention_hidden_dim)

    def forward(self, his_tokens: torch.Tensor, cand_tokens: torch.Tensor) -> torch.Tensor:
        """Training pass. his_tokens (B,H,T), cand_tokens (B,C,T) -> scores (B,C)."""
        user_vec = self.user_encoder(his_tokens)
        B, C, T = cand_tokens.shape
        news_vecs = self.news_encoder(cand_tokens.view(B * C, T)).view(B, C, -1)
        return torch.bmm(news_vecs, user_vec.unsqueeze(-1)).squeeze(-1)

    def score_single(self, his_tokens: torch.Tensor, cand_token: torch.Tensor) -> torch.Tensor:
        """Inference: his_tokens (B,H,T), cand_token (B,T) -> sigmoid scores (B,)."""
        user_vec = self.user_encoder(his_tokens)
        news_vec = self.news_encoder(cand_token)
        return torch.sigmoid((user_vec * news_vec).sum(dim=-1))


# =============================================================================
# 6. Datasets (Wu 2019 negative sampling mirrors sampling_strategy_wu2019)
# =============================================================================

class NRMSTrainDataset(Dataset):
    """Wu 2019 negative sampling: 1 positive + npratio random negatives."""

    def __init__(
        self,
        behaviors: ImpressionList,
        art_id_to_idx: Dict[str, int],
        article_matrix: np.ndarray,
        npratio: int = NPRATIO,
        history_size: int = HISTORY_SIZE,
        seed: int = 42,
    ):
        self.art_id_to_idx = art_id_to_idx
        self.article_matrix = article_matrix
        self.npratio = npratio
        self.history_size = history_size
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple] = []
        self._build(behaviors)

    def _lookup(self, aid) -> int:
        return self.art_id_to_idx.get(str(aid), 0)

    def _encode_history(self, history: List) -> np.ndarray:
        hist = list(history)[-self.history_size:]
        idxs = [self._lookup(aid) for aid in hist]
        idxs = [0] * (self.history_size - len(idxs)) + idxs
        return self.article_matrix[idxs]

    def _build(self, behaviors: ImpressionList) -> None:
        skipped = 0
        for imp in behaviors:
            candidates = imp.get("candidates", [])
            labels = imp.get("labels", [])
            history = imp.get("history", [])
            if not candidates or not labels:
                continue
            pos_ids = [aid for aid, lbl in zip(candidates, labels) if lbl == 1]
            neg_ids = [aid for aid, lbl in zip(candidates, labels) if lbl == 0]
            if not pos_ids or not neg_ids:
                skipped += 1
                continue
            his = self._encode_history(history)
            for pos_aid in pos_ids:
                neg_sample = self.rng.choice(neg_ids, size=self.npratio, replace=(len(neg_ids) < self.npratio)).tolist()
                cand_idxs = [self._lookup(pos_aid)] + [self._lookup(n) for n in neg_sample]
                cand_tokens = self.article_matrix[cand_idxs]
                self.samples.append((his, cand_tokens, np.int64(0)))
        log.info(f"NRMSTrainDataset: {len(self.samples):,} samples (skipped {skipped:,} impressions)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        his, cands, label = self.samples[idx]
        return torch.tensor(his, dtype=torch.long), torch.tensor(cands, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class NRMSEvalDataset(Dataset):
    """One candidate per sample (eval_mode=True in ebrec NRMSDataLoader)."""

    def __init__(
        self,
        behaviors: ImpressionList,
        art_id_to_idx: Dict[str, int],
        article_matrix: np.ndarray,
        history_size: int = HISTORY_SIZE,
    ):
        self.art_id_to_idx = art_id_to_idx
        self.article_matrix = article_matrix
        self.history_size = history_size
        self.samples: List[Tuple] = []
        self.impression_ids: List[int] = []
        self.labels_per_imp: List[List[int]] = []
        self.group_sizes: List[int] = []
        self._build(behaviors)

    def _lookup(self, aid) -> int:
        return self.art_id_to_idx.get(str(aid), 0)

    def _encode_history(self, history: List) -> np.ndarray:
        hist = list(history)[-self.history_size:]
        idxs = [self._lookup(aid) for aid in hist]
        idxs = [0] * (self.history_size - len(idxs)) + idxs
        return self.article_matrix[idxs]

    def _build(self, behaviors: ImpressionList) -> None:
        for imp in behaviors:
            candidates = imp.get("candidates", [])
            labels = imp.get("labels", [])
            history = imp.get("history", [])
            imp_id = imp.get("impression_id", 0)
            if not candidates:
                continue
            his = self._encode_history(history)
            for aid in candidates:
                self.samples.append((his, self.article_matrix[[self._lookup(aid)]]))
            self.impression_ids.append(imp_id)
            self.labels_per_imp.append(labels or [0] * len(candidates))
            self.group_sizes.append(len(candidates))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        his, cand = self.samples[idx]
        return torch.tensor(his, dtype=torch.long), torch.tensor(cand[0], dtype=torch.long)


# =============================================================================
# 7. Trainer
# =============================================================================

class NRMSTrainer:
    """
    Full NRMS training + evaluation pipeline.
    Mirrors the main loop in nrms_ebnerd.py (without TF/ebrec dependencies).
    """

    def __init__(
        self,
        dataset: str,
        epochs: int = 5,
        batch_size_train: int = 32,
        batch_size_eval: int = 256,
        npratio: int = NPRATIO,
        history_size: int = HISTORY_SIZE,
        title_size: int = TITLE_SIZE,
        head_num: int = HEAD_NUM,
        head_dim: int = HEAD_DIM,
        attention_hidden_dim: int = ATTENTION_HIDDEN_DIM,
        dropout: float = DROPOUT,
        lr: float = LR,
        max_vocab: int = 30_000,
        word_emb_dim: int = WORD_EMB_DIM,
        vocab_backend: str = "transformer",
        transformer_model: str = DEFAULT_TRANSFORMER_MODEL,
        device=None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.npratio = npratio
        self.history_size = history_size
        self.title_size = title_size
        self.max_vocab = max_vocab
        self.word_emb_dim = word_emb_dim
        self.vocab_backend = vocab_backend
        self.transformer_model = transformer_model
        self.device = device or DEVICE
        self.seed = seed
        self.model_hparams = dict(head_num=head_num, head_dim=head_dim, attention_hidden_dim=attention_hidden_dim, dropout=dropout)
        self.lr = lr
        self.model: Optional[NRMSModel] = None
        self.vocab: Optional[NRMSVocab] = None
        self.art_id_to_idx: Optional[Dict[str, int]] = None
        self.article_matrix: Optional[np.ndarray] = None
        self.pretrained_emb: Optional[np.ndarray] = None

    def prepare_data(self, articles: ArticleDict) -> None:
        # Text is "title subtitle body" (ebrec: concat_str_columns over
        # TEXT_COLUMNS_TO_USE), not the title alone.
        texts = [article_text(art) for art in articles.values()]
        self.vocab = NRMSVocab(
            max_vocab=self.max_vocab,
            title_size=self.title_size,
            backend=self.vocab_backend,
            model_name=self.transformer_model,
        ).build(texts)
        self.art_id_to_idx, self.article_matrix = build_article_title_matrix(articles, self.vocab)
        log.info(f"Article matrix: {self.article_matrix.shape} | Vocab: {self.vocab.vocab_size:,}")

        # Pretrained word-embedding matrix (ebrec: get_transformers_word_embeddings).
        if self.vocab_backend == "transformer":
            emb = load_transformer_word_embeddings(self.transformer_model)
            if emb.shape[0] < self.vocab.vocab_size:
                raise ValueError(
                    f"pretrained matrix has {emb.shape[0]} rows but the tokenizer "
                    f"vocab is {self.vocab.vocab_size}"
                )
            self.pretrained_emb = emb
            if self.word_emb_dim != emb.shape[1]:
                log.warning(
                    f"word_emb_dim={self.word_emb_dim} but the pretrained matrix is "
                    f"{emb.shape[1]}-d; adopting the pretrained dimension."
                )
                self.word_emb_dim = int(emb.shape[1])

    def build_model(self) -> NRMSModel:
        assert self.vocab is not None
        model = NRMSModel(
            vocab_size=self.vocab.vocab_size,
            word_emb_dim=self.word_emb_dim,
            pretrained_emb=self.pretrained_emb,
            **self.model_hparams,
        ).to(self.device)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"NRMS model: {n:,} params | device={self.device}")
        return model

    def train(self, articles: ArticleDict, train_behaviors: ImpressionList) -> "NRMSTrainer":
        self.prepare_data(articles)
        self.model = self.build_model()

        train_ds = NRMSTrainDataset(train_behaviors, self.art_id_to_idx, self.article_matrix, self.npratio, self.history_size, self.seed)
        if len(train_ds) == 0:
            log.warning("Empty training dataset!")
            return self

        loader = DataLoader(train_ds, batch_size=self.batch_size_train, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()  # matches categorical_crossentropy in Keras

        log.info(f"Training NRMS {self.epochs} epochs | {len(train_ds):,} samples | {len(loader):,} batches/epoch")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for his, cands, labels in loader:
                his, cands, labels = his.to(self.device), cands.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(his, cands), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            log.info(f"  Epoch {epoch:02d}/{self.epochs} | loss={total_loss/max(n_batches,1):.4f}")
        return self

    @torch.no_grad()
    def score_behaviors(self, behaviors: ImpressionList) -> Tuple[List[List[float]], List[List[int]], List[int]]:
        assert self.model is not None
        self.model.eval()
        eval_ds = NRMSEvalDataset(behaviors, self.art_id_to_idx, self.article_matrix, self.history_size)
        loader = DataLoader(eval_ds, batch_size=self.batch_size_eval, shuffle=False, num_workers=0)

        flat_scores: List[float] = []
        for his, cand in loader:
            flat_scores.extend(self.model.score_single(his.to(self.device), cand.to(self.device)).cpu().tolist())

        all_scores, all_labels, idx = [], [], 0
        for n, labels in zip(eval_ds.group_sizes, eval_ds.labels_per_imp):
            all_scores.append(flat_scores[idx:idx+n])
            all_labels.append(labels)
            idx += n
        return all_scores, all_labels, eval_ds.impression_ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "vocab_word2id": self.vocab.word2id,
                    "vocab_size": self.vocab.vocab_size, "art_id_to_idx": self.art_id_to_idx,
                    "article_matrix": self.article_matrix, "title_size": self.title_size,
                    "history_size": self.history_size, "model_hparams": self.model_hparams,
                    "word_emb_dim": self.word_emb_dim}, path)
        log.info(f"Saved NRMS -> {path}")

    def load(self, path: Path) -> "NRMSTrainer":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.vocab = NRMSVocab(title_size=ckpt["title_size"])
        self.vocab.word2id = ckpt["vocab_word2id"]
        self.vocab.vocab_size = ckpt["vocab_size"]
        self.art_id_to_idx = ckpt["art_id_to_idx"]
        self.article_matrix = ckpt["article_matrix"]
        self.title_size = ckpt["title_size"]
        self.history_size = ckpt["history_size"]
        self.model_hparams = ckpt["model_hparams"]
        self.word_emb_dim = ckpt.get("word_emb_dim", WORD_EMB_DIM)
        self.model = NRMSModel(vocab_size=self.vocab.vocab_size, word_emb_dim=self.word_emb_dim, **self.model_hparams).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        log.info(f"Loaded NRMS from {path}")
        return self


# =============================================================================
# 8. End-to-End Runner
# =============================================================================

def run_nrms_baseline(
    dataset: str,
    epochs: int = 5,
    sample_size: Optional[int] = None,
    train_sample_size: Optional[int] = None,
    batch_size_train: int = 32,
    batch_size_eval: int = 256,
) -> Dict[str, dict]:
    """End-to-end NRMS training + sliced evaluation. Matches nrms_ebnerd.py workflow."""
    log.info("=" * 70)
    log.info(f"  NRMS Baseline (PyTorch, no ebrec) -- {dataset.upper()}")
    log.info(f"  Epochs={epochs} | npratio={NPRATIO} | hist={HISTORY_SIZE} | title={TITLE_SIZE}")
    log.info("=" * 70)

    articles = load_articles(dataset, split="val")

    # nrms_ebnerd.py uses the validation split as training data (PATH/validation/)
    # For MIND we use the train split
    if dataset == "ebnerd":
        train_behaviors = load_behaviors(dataset, split="val", sample_size=train_sample_size or 5000)
    else:
        try:
            train_behaviors = load_behaviors(dataset, split="train", sample_size=train_sample_size or 5000)
        except Exception:
            train_behaviors = load_behaviors(dataset, split="val", sample_size=train_sample_size or 5000)

    val_behaviors = load_behaviors(dataset, split="val", sample_size=sample_size or 2000)
    val_behaviors = [imp for imp in val_behaviors if len(imp.get("candidates", [])) > 0 and sum(imp.get("labels", []) or []) > 0]
    log.info(f"  Train: {len(train_behaviors):,} impressions | Val: {len(val_behaviors):,} evaluable impressions")

    trainer = NRMSTrainer(dataset=dataset, epochs=epochs, batch_size_train=batch_size_train, batch_size_eval=batch_size_eval, device=DEVICE)

    model_path = PATHS["models"] / f"nrms_{dataset}.pth"
    if model_path.exists():
        log.info(f"Loading checkpoint from {model_path}")
        trainer.prepare_data(articles)
        trainer.model = trainer.build_model()
        trainer.load(model_path)
    else:
        trainer.train(articles, train_behaviors)
        trainer.save(model_path)

    log.info("Scoring validation split ...")
    all_scores, all_labels, imp_ids = trainer.score_behaviors(val_behaviors)

    for imp, labels in zip(val_behaviors, all_labels):
        imp["labels"] = labels

    log.info("Running full evaluation ...")
    slices_results = evaluate_all_slices(impressions_data=val_behaviors, all_scores=all_scores, articles=articles, n_bootstrap=1000)
    md_table = format_results_markdown(results=slices_results, dataset_name=dataset, model_name=f"NRMS-Baseline (PyTorch, {epochs}ep)")
    print("\n" + md_table + "\n")
    log.info("\n" + md_table)
    return slices_results


# =============================================================================
# 9. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NRMS Baseline -- PyTorch reproduction of ebnerd-benchmark")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="ebnerd")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=None, help="Val impressions (None=all)")
    parser.add_argument("--train_sample_size", type=int, default=None, help="Train impressions (None=all)")
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    if args.force_retrain:
        p = PATHS["models"] / f"nrms_{args.dataset}.pth"
        if p.exists():
            p.unlink()
            log.info(f"Deleted checkpoint {p}")

    run_nrms_baseline(dataset=args.dataset, epochs=args.epochs, sample_size=args.sample_size,
                      train_sample_size=args.train_sample_size, batch_size_train=args.batch_size_train,
                      batch_size_eval=args.batch_size_eval)


if __name__ == "__main__":
    main()
