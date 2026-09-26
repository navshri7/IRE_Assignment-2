"""
src/reranker.py — LightGBM LambdaRank & PyTorch MLP re-rankers.

Assignment 2: News Recommendation System
IRE CS4.406

Models:
 1. LightGBMReranker:
    - LGBMRanker with 'lambdarank' objective
    - Grouped listwise ranking on impression candidate sets
    - Early stopping on validation NDCG@5
    - Feature importance extraction

 2. MLPReranker:
    - 4-layer PyTorch neural network: Linear(20→256→128→64→1)
    - BatchNorm1d + Dropout + ReLU activations
    - Class-imbalance-weighted BCEWithLogitsLoss (pos_weight = neg/pos)
    - Adam optimizer + CosineAnnealingLR
    - Device-aware (CUDA, MPS, CPU)

CLI USAGE:
  .venv/bin/python3 src/reranker.py --model lgbm --dataset ebnerd --sample_size 1000
  .venv/bin/python3 src/reranker.py --model mlp  --dataset mind   --sample_size 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is in sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import pickle
import sys

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import DEVICE, FEATURE_NAMES, HYPERPARAMS, N_FEATURES, PATHS
from src.feature_engineering import build_and_save_features, extract_features_dataset, FeatureExtractor
from src.data_loader import load_articles, load_behaviors
from src.logger import get_logger

log = get_logger(__name__)
LGBM_N_JOBS = 1 if sys.platform == "darwin" else -1


# ─────────────────────────────────────────────────────────────────────────────
# 1. LightGBM LambdaRank Re-Ranker
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMReranker:
    """
    Listwise LambdaRank re-ranker using LightGBM.
    Takes (X, y, groups) where groups specify candidate set sizes per impression.
    """
    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        min_child_samples: int = 10,
        early_stopping: int = 50,
        n_jobs: int = None,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.early_stopping = early_stopping
        # See LGBM_N_JOBS: n_jobs=-1 segfaults on macOS once torch/MPS is live.
        self.n_jobs = LGBM_N_JOBS if n_jobs is None else n_jobs
        self.model: Optional[lgb.LGBMRanker] = None
        self.booster: Optional[lgb.Booster] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> "LightGBMReranker":
        """Trains the LightGBM LambdaRank model."""
        log.info(f"Training LightGBMReranker on {X_train.shape[0]:,} pairs ({len(groups_train):,} impressions) ...")

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            importance_type="gain",
            verbose=-1,
            random_state=42,
        )

        eval_set = None
        eval_group = None
        callbacks = []

        if X_val is not None and y_val is not None and groups_val is not None and len(groups_val) > 0:
            eval_set = [(X_val, y_val)]
            eval_group = [groups_val]
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping, verbose=False))

        self.model.fit(
            X_train,
            y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_metric="ndcg",
            eval_at=[5, 10],
            callbacks=callbacks,
        )

        self.booster = self.model.booster_
        best_iter = getattr(self.booster, "best_iteration", self.n_estimators)
        log.info(f"LightGBM fit complete. Best iteration: {best_iter}")
        return self

    def predict_scores(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Predicts ranking scores for feature matrix X."""
        if self.booster is None and self.model is None:
            raise RuntimeError("Model is not fitted or loaded yet.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)

        predictor = self.model if self.model is not None else self.booster
        scores = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            s = predictor.predict(batch)
            scores.append(s)
        return np.concatenate(scores).astype(np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns gain-based feature importances."""
        if self.booster is not None:
            importances = self.booster.feature_importance(importance_type="gain")
            return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}
        if self.model is not None:
            importances = self.model.feature_importances_
            return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}
        return {}

    def save(self, path: Path, fingerprint: Optional[dict] = None) -> None:
        """
        Saves LightGBM model to file.

        fingerprint: optional dict describing the feature *semantics* the model
        was trained under (e.g. {"dense_backend": "minilm"}). Stored as JSON
        sidecar so a later load can refuse a semantically incompatible
        checkpoint. A count match alone is NOT sufficient: switching the dense
        backend keeps N_FEATURES at 28 while completely changing the meaning of
        the `dense_score` and `hybrid_score` columns.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.booster is not None:
            self.booster.save_model(str(path))
        elif self.model is not None and hasattr(self.model, "booster_"):
            self.model.booster_.save_model(str(path))
        else:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

        if fingerprint is not None:
            meta = dict(fingerprint)
            meta["n_features"] = N_FEATURES
            meta["feature_names"] = list(FEATURE_NAMES)
            path.with_suffix(path.suffix + ".meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        log.info(f"Saved LightGBM model to {path}")

    def load(self, path: Path, fingerprint: Optional[dict] = None) -> "LightGBMReranker":
        """
        Loads a LightGBM model, verifying BOTH the feature count and (when a
        fingerprint is supplied) the feature semantics it was trained under.
        """
        try:
            self.booster = lgb.Booster(model_file=str(path))
            self.model = None
        except Exception:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.booster = getattr(self.model, "booster_", None)

        # A checkpoint trained on the original 20-feature set cannot score the
        # 28-feature matrix (8 session columns were added in Q1.2). Fail loudly
        # rather than silently scoring garbage or raising an opaque shape error.
        expected = None
        if self.booster is not None:
            try:
                expected = int(self.booster.num_feature())
            except Exception:
                expected = None
        if expected is not None and expected != N_FEATURES:
            raise ValueError(
                f"LightGBM checkpoint {path.name} was trained on {expected} features but "
                f"the pipeline now builds {N_FEATURES} (FEATURE_NAMES). Retrain the "
                f"re-ranker, or set N_FEATURES to match the checkpoint."
            )

        # Semantic check: same column count, different meaning.
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if fingerprint is not None:
            if not meta_path.exists():
                # A checkpoint written before fingerprints existed cannot be
                # verified, so it must not be trusted: it may well predate the
                # current dense backend. Treat as stale rather than guessing.
                raise ValueError(
                    f"LightGBM checkpoint {path.name} has no .meta.json sidecar, so its "
                    f"feature schema is unverifiable. It may predate the current dense "
                    f"backend, which would score silently-wrong values. Retrain, or pass "
                    f"fingerprint=None to accept it at your own risk."
                )
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            missing = sorted(k for k in fingerprint if k not in meta)
            mismatched = {
                k: (meta.get(k), v)
                for k, v in fingerprint.items()
                if k in meta and meta.get(k) != v
            }
            if missing or mismatched:
                details = ", ".join(
                    [f"{k}: <absent from sidecar>" for k in missing]
                    + [
                        f"{k}: checkpoint={old!r} current={new!r}"
                        for k, (old, new) in mismatched.items()
                    ]
                )
                raise ValueError(
                    f"LightGBM checkpoint {path.name} was trained under a different "
                    f"feature schema ({details}). The column count still matches "
                    f"{N_FEATURES}, so this would otherwise score silently-wrong values. "
                    f"Retrain, or pass fingerprint=None to accept it at your own risk."
                )

        log.info(f"Loaded LightGBM model from {path} ({expected} features)")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch MLP Re-Ranker
# ─────────────────────────────────────────────────────────────────────────────

class PyTorchMLP(nn.Module):
    """
    3-hidden-layer MLP architecture: (20 → 256 → 128 → 64 → 1).
    """
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rates: List[float] = [0.3, 0.2, 0.0],
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if drop > 0.0:
                layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        # Final output logit
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPReranker:
    """
    PyTorch MLP ranker with weighted BCE loss and early stopping.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        patience: int = 5,
        batch_size: int = 2048,
        device: Optional[torch.device] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or HYPERPARAMS.get("mlp_hidden_dims", [256, 128, 64])
        self.dropout_rates = dropout_rates or HYPERPARAMS.get("mlp_dropout", [0.3, 0.2, 0.0])
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device or DEVICE

        self.model = PyTorchMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPReranker":
        """Trains the PyTorch MLP on feature matrix with early stopping on val AUC."""
        pos_count = max(1, int((y_train == 1).sum()))
        neg_count = max(1, int((y_train == 0).sum()))
        pos_weight_val = neg_count / pos_count
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)

        log.info(
            f"Training MLPReranker on {self.device} ({len(X_train):,} pairs, pos_weight={pos_weight_val:.2f}) ..."
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        best_val_auc = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)

            # Validation step
            if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
                val_scores = self.predict_scores(X_val)
                try:
                    val_auc = float(roc_auc_score(y_val, val_scores))
                except Exception:
                    val_auc = 0.5

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                log.info(
                    f"Epoch {epoch:02d}/{self.epochs:02d} | train_loss: {avg_loss:.4f} | val_auc: {val_auc:.4f} (best: {best_val_auc:.4f})"
                )

                if patience_counter >= self.patience:
                    log.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                log.info(f"Epoch {epoch:02d}/{self.epochs:02d} | train_loss: {avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info(f"Restored best model with val_auc: {best_val_auc:.4f}")

        return self

    def predict_scores(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Predicts probability scores for feature matrix X."""
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)

        self.model.eval()
        scores: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                scores.append(probs)

        return np.concatenate(scores).astype(np.float32)

    def save(self, path: Path, fingerprint: Optional[dict] = None) -> None:
        """
        Saves PyTorch model state dict.

        fingerprint: written to a sidecar .meta.json describing the feature
        SEMANTICS the model was trained under. The MLP's input layer width is
        N_FEATURES either way, so a checkpoint trained under a different dense
        backend has an identical shape and would load without complaint while
        scoring silently-wrong values -- exactly the failure the LightGBM
        fingerprint already guards against.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))
        if fingerprint is not None:
            meta = dict(fingerprint)
            meta["n_features"] = N_FEATURES
            meta["feature_names"] = list(FEATURE_NAMES)
            path.with_suffix(path.suffix + ".meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        log.info(f"Saved PyTorch MLP model to {path}")

    def load(self, path: Path, fingerprint: Optional[dict] = None) -> "MLPReranker":
        """Loads PyTorch state dict, verifying feature count and semantics."""
        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state)

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if fingerprint is not None:
            if not meta_path.exists():
                raise RuntimeError(
                    f"MLP checkpoint {path.name} has no .meta.json sidecar, so its "
                    f"feature schema is unverifiable. It may predate the current dense "
                    f"backend, which would score silently-wrong values. Retrain, or "
                    f"pass fingerprint=None to accept it at your own risk."
                )
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            # Strict superset check -- see the equivalent comment in
            # LightGBMReranker.load. A key absent from the sidecar is a mismatch,
            # not a pass.
            missing = sorted(k for k in fingerprint if k not in meta)
            mismatched = {
                k: (meta.get(k), v)
                for k, v in fingerprint.items()
                if k in meta and meta.get(k) != v
            }
            if missing or mismatched:
                details = ", ".join(
                    [f"{k}: <absent from sidecar>" for k in missing]
                    + [
                        f"{k}: checkpoint={old!r} current={new!r}"
                        for k, (old, new) in mismatched.items()
                    ]
                )
                raise RuntimeError(
                    f"MLP checkpoint {path.name} was trained under a different "
                    f"feature schema ({details}). The input width still matches "
                    f"{N_FEATURES}, so this would otherwise score silently-wrong "
                    f"values. Retrain, or pass fingerprint=None to accept it at your "
                    f"own risk."
                )
        log.info(f"Loaded PyTorch MLP model from {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Re-Ranker training & evaluation")
    parser.add_argument("--model", choices=["lgbm", "mlp"], default="lgbm", help="Model type to train")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="ebnerd", help="Dataset to train on")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of impressions for train/val")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for MLP")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"  Phase 4: Re-Ranker — {args.model.upper()} on {args.dataset.upper()}")
    log.info("=" * 60)

    # 1. Load data & extract features for train and val splits
    log.info(f"Extracting features for {args.dataset} [train] (sample={args.sample_size}) ...")
    articles_train = load_articles(args.dataset, split="train")
    behaviors_train = load_behaviors(args.dataset, split="train", sample_size=args.sample_size)
    extractor_train = FeatureExtractor(articles_train)
    X_tr, y_tr, grp_tr, _, _ = extract_features_dataset(extractor_train, behaviors_train)

    log.info(f"Extracting features for {args.dataset} [val] (sample={args.sample_size // 2}) ...")
    articles_val = load_articles(args.dataset, split="val")
    behaviors_val = load_behaviors(args.dataset, split="val", sample_size=max(50, args.sample_size // 2))
    extractor_val = FeatureExtractor(articles_val)
    X_va, y_va, grp_va, _, _ = extract_features_dataset(extractor_val, behaviors_val)

    # 2. Train and evaluate selected model
    models_dir = PATHS["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "lgbm":
        reranker = LightGBMReranker(
            n_estimators=HYPERPARAMS.get("lgbm_n_estimators", 300),
            num_leaves=HYPERPARAMS.get("lgbm_num_leaves", 63),
            learning_rate=HYPERPARAMS.get("lgbm_learning_rate", 0.05),
            early_stopping=HYPERPARAMS.get("lgbm_early_stopping", 30),
        )
        reranker.fit(X_tr, y_tr, grp_tr, X_val=X_va, y_val=y_va, groups_val=grp_va)

        # Print top feature importances
        importances = reranker.get_feature_importances()
        log.info("Top 10 Feature Importances (Gain):")
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, val in sorted_imp:
            log.info(f"  {name:<24}: {val:.2f}")

        # Save model
        save_path = models_dir / f"lgbm_{args.dataset}.txt"
        reranker.save(save_path)

    elif args.model == "mlp":
        reranker = MLPReranker(
            epochs=args.epochs,
            lr=HYPERPARAMS.get("mlp_lr", 1e-3),
            batch_size=HYPERPARAMS.get("mlp_batch_size", 2048),
        )
        reranker.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        # Save model
        save_path = models_dir / f"mlp_{args.dataset}.pth"
        reranker.save(save_path)

    # Sanity prediction
    val_preds = reranker.predict_scores(X_va[:100])
    log.info(f"Sample validation predictions (first 5): {[round(float(p), 4) for p in val_preds[:5]]}")
    log.info("Phase 4 re-ranker check complete.")


if __name__ == "__main__":
    main()
