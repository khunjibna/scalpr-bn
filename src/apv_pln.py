"""APV-PLN — Adaptive Price-Volume Probabilistic Learner Network (handbook §APV-PLN).

Archetype VII — Probabilistic Trend & Regime Learner.
Trained via Privileged Information Distillation (Oracle Teacher / LUPI).

PyTorch is imported lazily so the rest of the bot continues to run even when
torch isn't installed.  Use :func:`is_available` to gate any APV-PLN call.

Components (all PyTorch):
    PriceCNN   : 2× Conv1D + LayerNorm + LeakyReLU                 over [B, 32, 5]
    VolumeCNN  : 2× Conv1D + LayerNorm + LeakyReLU                 over [B, 32, 5]
    Cross-Attn : price ↔ volume cross-attention (single-head MHA)
    Gate       : adaptive sigmoid gate combining fused & residual paths
    Head       : Linear → 51 logits (bin-distribution over forward returns)

Oracle Teacher (train only):
    ManualLSTM over 5 future bars (2-dim each) → 51-bin softmax
Loss:
    L = α·CE(student, y_bin) + β·T²·KL(student/T || oracle/T)
    α = β = 0.5, T = 2.0
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger

from .interfaces import ModelInterface, Signal

# ─── Lazy torch import ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except Exception as _e:                # pragma: no cover
    torch = None                       # type: ignore
    nn = None                          # type: ignore
    F = None                           # type: ignore
    DataLoader = TensorDataset = None  # type: ignore
    _TORCH_OK = False
    _TORCH_ERR = _e


def is_available() -> bool:
    """True if PyTorch is importable in the current environment."""
    return _TORCH_OK


def resolve_device(backend: str = "cpu"):
    """Resolve a torch device string ('cpu', 'cuda', 'directml')."""
    if not _TORCH_OK:
        return "cpu"
    if backend == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if backend == "directml":
        try:
            import torch_directml  # type: ignore
            return torch_directml.device()
        except Exception:
            logger.warning("torch_directml not installed — falling back to CPU")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Bin grid (handbook: 51 bins covering 0.5–99.5th pct of training fwd returns)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_NUM_BINS = 51


def compute_bin_centres(returns: np.ndarray,
                        num_bins: int = DEFAULT_NUM_BINS,
                        lo_pct: float = 0.5,
                        hi_pct: float = 99.5) -> np.ndarray:
    """Return ``num_bins`` evenly-spaced bin centres clamped to the given
    percentile bounds of the training forward-return distribution."""
    lo = float(np.percentile(returns, lo_pct))
    hi = float(np.percentile(returns, hi_pct))
    if hi <= lo:                                  # degenerate (constant returns)
        hi = lo + 1e-6
    edges  = np.linspace(lo, hi, num_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres.astype(np.float32)


def returns_to_bins(returns: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Bucket each return into the closest bin index in ``centres``."""
    # Vectorised nearest-bin lookup
    idx = np.searchsorted(centres, returns, side="left")
    idx = np.clip(idx, 0, len(centres) - 1)
    # Refine: also check the bin to the left
    left = np.maximum(idx - 1, 0)
    pick_left = np.abs(returns - centres[left]) < np.abs(returns - centres[idx])
    idx[pick_left] = left[pick_left]
    return idx.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Model — built only when torch is available
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:

    class _Conv1DStack(nn.Module):
        """2× (Conv1D → LayerNorm → LeakyReLU) on a [B, seq, feat] tensor.
        Returns [B, seq, channels]."""

        def __init__(self, in_feat: int, channels: int, dropout: float = 0.15):
            super().__init__()
            self.conv1 = nn.Conv1d(in_feat,  channels, kernel_size=3, padding=1)
            self.ln1   = nn.LayerNorm(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.ln2   = nn.LayerNorm(channels)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, seq, feat]  →  [B, feat, seq] for Conv1D
            h = x.transpose(1, 2)
            h = self.conv1(h)
            h = h.transpose(1, 2)
            h = F.leaky_relu(self.ln1(h), 0.1)
            h = self.drop(h)
            h = h.transpose(1, 2)
            h = self.conv2(h)
            h = h.transpose(1, 2)
            h = F.leaky_relu(self.ln2(h), 0.1)
            return h


    class APVPLNNet(nn.Module):
        """Adaptive Price-Volume Probabilistic Learner Network."""

        def __init__(self,
                     price_feat:  int = 5,
                     volume_feat: int = 5,
                     seq_len:     int = 32,
                     cnn_channels: int = 64,
                     nhead:       int = 4,
                     dropout:     float = 0.15,
                     num_bins:    int = DEFAULT_NUM_BINS):
            super().__init__()
            self.seq_len  = seq_len
            self.num_bins = num_bins

            self.price_cnn  = _Conv1DStack(price_feat,  cnn_channels, dropout)
            self.volume_cnn = _Conv1DStack(volume_feat, cnn_channels, dropout)

            # Cross-attention: query=price stream, key/value=volume stream
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=cnn_channels, num_heads=nhead,
                dropout=dropout, batch_first=True,
            )

            # Adaptive gate: σ(linear([fused; residual])) ⊙ fused
            self.gate = nn.Sequential(
                nn.Linear(cnn_channels * 2, cnn_channels),
                nn.Sigmoid(),
            )

            # Pool over time → 51-bin head
            self.head = nn.Sequential(
                nn.LayerNorm(cnn_channels),
                nn.Linear(cnn_channels, cnn_channels),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(cnn_channels, num_bins),
            )

        def forward(self, x_price, x_volume):
            p = self.price_cnn(x_price)    # [B, seq, C]
            v = self.volume_cnn(x_volume)  # [B, seq, C]
            fused, _ = self.cross_attn(p, v, v)
            gate_in  = torch.cat([fused, p], dim=-1)
            gated    = self.gate(gate_in) * fused + (1 - self.gate(gate_in)) * p
            pooled   = gated.mean(dim=1)   # mean-pool over time
            return self.head(pooled)       # [B, num_bins] logits


    class OracleTeacher(nn.Module):
        """ManualLSTM over 5 future bars (privileged info, train only)."""

        def __init__(self, in_feat: int = 2, hidden: int = 32,
                     num_bins: int = DEFAULT_NUM_BINS):
            super().__init__()
            self.lstm = nn.LSTM(in_feat, hidden, batch_first=True)
            self.head = nn.Linear(hidden, num_bins)

        def forward(self, x_oracle):
            # x_oracle: [B, horizon=5, 2]
            _, (h, _) = self.lstm(x_oracle)
            return self.head(h.squeeze(0))   # [B, num_bins] logits

else:
    APVPLNNet    = None  # type: ignore
    OracleTeacher = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Public model wrapper — implements ModelInterface (§14.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APVPLNHParams:
    cnn_channels: int   = 64
    nhead:        int   = 4
    dropout:      float = 0.15
    num_bins:     int   = DEFAULT_NUM_BINS
    seq_len:      int   = 32
    horizon:      int   = 5
    lr:           float = 1e-3
    batch_size:   int   = 64
    epochs:       int   = 20
    alpha:        float = 0.5     # CE weight
    beta:         float = 0.5     # KL weight
    temperature:  float = 2.0     # distillation T


class APVPLNModel(ModelInterface):
    """Inference-side wrapper conforming to ModelInterface.

    Training is intentionally kept as a class-method (:meth:`train`) so the
    online bot does not pull torch into hot loops.
    """

    model_version = "APV_PLN_v1"
    backend       = "cpu"

    def __init__(self, weights_path: str, config: Optional[dict] = None,
                 backend: str = "cpu"):
        if not _TORCH_OK:
            raise RuntimeError(
                f"APV-PLN requires PyTorch (install with `pip install torch`). "
                f"Original import error: {_TORCH_ERR}"
            )
        self.config = config or {}
        self.backend = backend
        self.device  = resolve_device(backend)
        self._load(weights_path)

    # ── load / save ──────────────────────────────────────────────────────────

    def _load(self, weights_path: str):
        payload = torch.load(weights_path, map_location=self.device, weights_only=False)
        hp = payload.get("hparams", APVPLNHParams().__dict__)
        self.hparams      = APVPLNHParams(**hp) if isinstance(hp, dict) else hp
        self.bin_centres  = payload["bin_centres"]      # np.ndarray (num_bins,)
        self.price_scaler = payload.get("price_scaler")  # sklearn-like or None
        self.volume_scaler = payload.get("volume_scaler")
        self.model_version = payload.get("model_version", self.model_version)

        net = APVPLNNet(
            cnn_channels=self.hparams.cnn_channels,
            nhead=self.hparams.nhead,
            dropout=self.hparams.dropout,
            num_bins=self.hparams.num_bins,
            seq_len=self.hparams.seq_len,
        )
        net.load_state_dict(payload["state_dict"])
        net.to(self.device).eval()
        self.net = net
        logger.info(
            f"APV-PLN loaded | version={self.model_version} | device={self.device} | "
            f"params={sum(p.numel() for p in net.parameters()):,}"
        )

    @staticmethod
    def save(path: str, net: "APVPLNNet", hparams: APVPLNHParams,
             bin_centres: np.ndarray, model_version: str,
             price_scaler=None, volume_scaler=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict":    net.state_dict(),
            "hparams":       hparams.__dict__,
            "bin_centres":   bin_centres,
            "price_scaler":  price_scaler,
            "volume_scaler": volume_scaler,
            "model_version": model_version,
        }, path)
        logger.info(f"APV-PLN saved → {path}")

    # ── inference (ModelInterface) ───────────────────────────────────────────

    def predict(self, features: np.ndarray) -> dict:
        """``features`` is a dict-like packed array — but the interface forces
        ``np.ndarray``.  To stay schema-compliant we accept either:

        * ``np.ndarray`` of shape (seq_len, price_feat + volume_feat)
        * ``np.ndarray`` of shape (B, seq_len, price_feat + volume_feat)
        """
        if not _TORCH_OK:
            raise RuntimeError("APV-PLN inference requires PyTorch")

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        seq_len = self.hparams.seq_len
        pf      = 5  # price_feat
        if x.shape[1] != seq_len or x.shape[2] != pf * 2:
            raise ValueError(
                f"APV-PLN expects (B, {seq_len}, {pf*2}); got {x.shape}"
            )

        x_price  = torch.from_numpy(x[:, :, :pf]).to(self.device)
        x_volume = torch.from_numpy(x[:, :, pf:]).to(self.device)

        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.net(x_price, x_volume)   # [B, num_bins]
            probs  = F.softmax(logits, dim=-1)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs_np  = probs.cpu().numpy()
        centres   = self.bin_centres
        exp_ret   = (probs_np * centres).sum(axis=-1)              # expected return
        confidence = probs_np.max(axis=-1)                         # peak prob

        return {
            "prediction": exp_ret,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "version":    self.model_version,
            # extras (not in the strict §14.1 schema but handy)
            "probs":      probs_np,
            "bins":       centres,
        }

    def to_signal(self, symbol: str, features: np.ndarray,
                  archetype: str = "apv_pln") -> Signal:
        out = self.predict(features)
        exp_ret    = float(np.atleast_1d(out["prediction"])[0])
        confidence = float(np.atleast_1d(out["confidence"])[0])
        direction  = 1.0 if exp_ret > 0 else (-1.0 if exp_ret < 0 else 0.0)
        return Signal(
            timestamp=datetime.now(timezone.utc),
            archetype=archetype, symbol=symbol,
            direction=direction, confidence=confidence,
            model_name="APV_PLN", model_version=self.model_version,
            ensemble_size=1, ensemble_entropy=0.0,
            features_version=self.config.get("features_version", "2.2"),
            backend=str(self.device), inference_latency_ms=out["latency_ms"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training routine — LUPI distillation
# ─────────────────────────────────────────────────────────────────────────────

def train_apv_pln(
    X_price:  np.ndarray,    # [N, seq_len, 5]
    X_volume: np.ndarray,    # [N, seq_len, 5]
    X_oracle: np.ndarray,    # [N, horizon, 2]    — privileged, TRAIN ONLY
    fwd_returns: np.ndarray, # [N,]               — realised forward log-returns
    *,
    val_fraction: float = 0.2,
    hparams: Optional[APVPLNHParams] = None,
    save_path: Optional[str] = None,
    backend: str = "cpu",
    model_version: str = "APV_PLN_v1",
) -> dict:
    """Train an APV-PLN model with Oracle-teacher distillation.

    Oracle isolation contract (handbook):
        - train phase : oracle CALLED   → α·CE + β·T²·KL
        - val   phase : oracle NEVER    → CE only
    """
    if not _TORCH_OK:
        raise RuntimeError(
            f"PyTorch required to train APV-PLN. Original import error: {_TORCH_ERR}"
        )

    hp = hparams or APVPLNHParams()
    device = resolve_device(backend)

    # ── Build bin grid ───────────────────────────────────────────────────────
    bin_centres = compute_bin_centres(fwd_returns, num_bins=hp.num_bins)
    y_bin       = returns_to_bins(fwd_returns, bin_centres)

    # ── Chronological train/val split ────────────────────────────────────────
    n_val   = int(len(fwd_returns) * val_fraction)
    n_train = len(fwd_returns) - n_val
    sl_tr, sl_va = slice(0, n_train), slice(n_train, None)

    def _t(arr, dtype):
        return torch.tensor(arr, dtype=dtype, device=device)

    tr = TensorDataset(
        _t(X_price[sl_tr],  torch.float32),
        _t(X_volume[sl_tr], torch.float32),
        _t(X_oracle[sl_tr], torch.float32),
        _t(y_bin[sl_tr],    torch.long),
    )
    va = TensorDataset(
        _t(X_price[sl_va],  torch.float32),
        _t(X_volume[sl_va], torch.float32),
        _t(y_bin[sl_va],    torch.long),
    )
    tr_loader = DataLoader(tr, batch_size=hp.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=hp.batch_size, shuffle=False)

    # ── Models ───────────────────────────────────────────────────────────────
    student = APVPLNNet(
        cnn_channels=hp.cnn_channels, nhead=hp.nhead, dropout=hp.dropout,
        num_bins=hp.num_bins, seq_len=hp.seq_len,
    ).to(device)
    oracle = OracleTeacher(in_feat=2, hidden=32, num_bins=hp.num_bins).to(device)

    opt = torch.optim.AdamW(
        list(student.parameters()) + list(oracle.parameters()),
        lr=hp.lr, weight_decay=1e-4,
    )
    ce = nn.CrossEntropyLoss()
    T  = hp.temperature

    history = {"train_loss": [], "val_loss": [], "val_dir_acc": []}
    best_val = math.inf
    best_state = None

    for epoch in range(hp.epochs):
        # ── train ────────────────────────────────────────────────────────────
        student.train(); oracle.train()
        tr_loss_sum = 0.0; n_tr = 0
        for xp, xv, xo, yb in tr_loader:
            opt.zero_grad()
            s_logits = student(xp, xv)
            o_logits = oracle(xo)
            loss_ce = ce(s_logits, yb)
            loss_kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(o_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = hp.alpha * loss_ce + hp.beta * loss_kl
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(yb); n_tr += len(yb)
        tr_loss = tr_loss_sum / max(n_tr, 1)

        # ── val (Oracle NEVER called) ────────────────────────────────────────
        student.eval()
        va_loss_sum = 0.0; n_va = 0; dir_correct = 0
        with torch.no_grad():
            for xp, xv, yb in va_loader:
                s_logits = student(xp, xv)
                va_loss_sum += ce(s_logits, yb).item() * len(yb)
                n_va += len(yb)
                # Expected return direction vs true bin direction
                probs    = F.softmax(s_logits, dim=-1).cpu().numpy()
                exp_ret  = (probs * bin_centres).sum(axis=-1)
                true_ret = bin_centres[yb.cpu().numpy()]
                dir_correct += int(np.sum(np.sign(exp_ret) == np.sign(true_ret)))
        va_loss = va_loss_sum / max(n_va, 1)
        va_acc  = dir_correct / max(n_va, 1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_dir_acc"].append(va_acc)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        logger.info(
            f"APV-PLN epoch {epoch+1:02d}/{hp.epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dir_acc={va_acc:.3f}"
        )

    # ── Restore best & save ──────────────────────────────────────────────────
    if best_state is not None:
        student.load_state_dict(best_state)

    if save_path:
        APVPLNModel.save(save_path, student, hp, bin_centres, model_version)

    return {
        "hparams":      hp.__dict__,
        "bin_centres":  bin_centres,
        "history":      history,
        "best_val_loss": best_val,
        "model_version": model_version,
    }
