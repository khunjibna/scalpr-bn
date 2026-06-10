"""Ensemble system — 3 diverse classifiers (handbook §9).

Members:
    - RandomForest (bagging, deep trees)
    - GradientBoosting (sequential, shallow trees)
    - ExtraTrees (extra randomisation)

Weighted by validation Sharpe (softmax). Predictions return:
    (signal, confidence, agreement_ratio, size_scalar)
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .interfaces import ModelInterface

MEMBER_NAMES = ("rf", "gbm", "et")


def _safe_fit(model, X, y):
    """Fit while suppressing sklearn parallel UserWarning leaks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X, y)
    return model


class EnsembleModel(ModelInterface):
    """Weighted vote of 3 tree-based classifiers."""

    model_version: str = "ensemble_v1"
    backend:       str = "sklearn"

    def __init__(self):
        self.config: dict = {"features_version": "2.2"}
        self.scaler: StandardScaler | None = None
        self.members: dict = {}            # name → fitted estimator
        self.weights: dict[str, float] = {n: 1 / 3 for n in MEMBER_NAMES}
        self.member_metrics: dict[str, dict] = {}

    # ── Construction ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_members(seed: int, hp_overrides: dict | None = None) -> dict:
        hp = hp_overrides or {}
        n_est = hp.get("n_estimators", 200)
        depth = hp.get("max_depth", 10)
        leaf  = hp.get("min_samples_leaf", 10)
        return {
            "rf":  RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                max_features=hp.get("max_features", "sqrt"),
                n_jobs=1, random_state=seed,
            ),
            "gbm": GradientBoostingClassifier(
                n_estimators=min(n_est, 150), max_depth=max(3, depth // 3),
                min_samples_leaf=leaf, learning_rate=0.05,
                random_state=seed,
            ),
            "et":  ExtraTreesClassifier(
                n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                max_features=hp.get("max_features", "sqrt"),
                n_jobs=1, random_state=seed,
            ),
        }

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val,
            seed: int = 42, hp_overrides: dict | None = None,
            pnl_fn=None, sharpe_fn=None) -> dict:
        """Fit all 3 members, compute val metrics, set softmax weights.
        Returns dict of per-member val metrics.
        """
        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train)
        Xva = self.scaler.transform(X_val)

        members = self._build_members(seed, hp_overrides)
        metrics: dict[str, dict] = {}
        sharpes: list[float] = []

        for name, model in members.items():
            try:
                _safe_fit(model, Xtr, y_train)
                pred  = model.predict(Xva)
                acc   = float(accuracy_score(y_val, pred))
                pnl   = pnl_fn(y_val, pred) if pnl_fn else (pred == y_val).astype(float) - 0.5
                sharp = float(sharpe_fn(pnl)) if sharpe_fn else 0.0
                metrics[name] = {"acc": acc, "sharpe": sharp}
                sharpes.append(sharp)
                self.members[name] = model
                logger.info(f"Ensemble[{name}] acc={acc:.3f} sharpe={sharp:.3f}")
            except Exception as e:
                logger.warning(f"Ensemble[{name}] training failed: {e}")
                sharpes.append(0.0)

        # Softmax weights (temperature=1.0). If all sharpes ≤ 0 → equal weights.
        sh_arr = np.array(sharpes, dtype=float)
        if np.all(sh_arr <= 0):
            self.weights = {n: 1 / len(MEMBER_NAMES) for n in MEMBER_NAMES}
        else:
            sh_arr = np.clip(sh_arr, -2.0, 5.0)  # avoid overflow / extreme dominance
            exp = np.exp(sh_arr)
            w = exp / exp.sum()
            self.weights = {n: float(w[i]) for i, n in enumerate(MEMBER_NAMES)}

        self.member_metrics = metrics
        logger.info(
            f"Ensemble weights initialised: "
            + " ".join(f"{n}={self.weights[n]:.2f}" for n in MEMBER_NAMES)
        )
        return metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_proba_up(self, X) -> tuple[float, float, dict]:
        """Weighted probability of class=1 (up).
        Returns (weighted_prob_up, raw_probs_dict).
        """
        if self.scaler is None or not self.members:
            return 0.5, {n: 0.5 for n in MEMBER_NAMES}
        Xs = self.scaler.transform(X)
        probs: dict[str, float] = {}
        weighted = 0.0
        for name, model in self.members.items():
            try:
                p_up = float(model.predict_proba(Xs)[0, 1])
            except Exception:
                p_up = 0.5
            probs[name] = p_up
            weighted += self.weights.get(name, 0.0) * p_up
        return float(weighted), probs

    def predict_consensus(self, X, conf_threshold: float = 0.55,
                          min_agree: int = 2) -> tuple[int, float, float, float, dict]:
        """Generate ensemble signal with consensus + size scalar.

        Returns:
            signal:        +1 / -1 / 0
            confidence:    max(p_up, 1-p_up) of weighted average
            agreement:     fraction of members agreeing on direction
            size_scalar:   position-size multiplier in [0, 1] (handbook §9.3)
            raw_probs:     per-member prob_up dict
        """
        weighted_up, probs = self.predict_proba_up(X)
        p_down = 1.0 - weighted_up

        # Direction votes (per member)
        votes_up   = sum(1 for p in probs.values() if p > 0.5)
        votes_down = sum(1 for p in probs.values() if p < 0.5)
        total      = max(1, len(probs))
        agreement  = max(votes_up, votes_down) / total

        # Required consensus (§11.3): ≥ min_agree members agree on dominant side
        if weighted_up >= conf_threshold and votes_up >= min_agree:
            signal = 1
            confidence = weighted_up
        elif p_down >= conf_threshold and votes_down >= min_agree:
            signal = -1
            confidence = p_down
        else:
            signal = 0
            confidence = max(weighted_up, p_down)

        # Conflict-resolution size scalar (§9.3)
        if confidence > 0.70:
            size_scalar = 1.0
        elif confidence > 0.60:
            size_scalar = 0.7
        else:
            size_scalar = 0.3
        if abs(weighted_up - 0.5) < 0.10:  # weak signal magnitude
            size_scalar *= 0.5

        return signal, float(confidence), float(agreement), float(size_scalar), probs

    # ── ModelInterface (§14.1) ───────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> dict:
        """Standardised ModelInterface inference returning the §14.1 schema.

        Use :meth:`predict_consensus` when you need the (signal, conf,
        agreement, size_scalar, probs) tuple consumed by MLStrategy / Trader.
        """
        import time
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        t0 = time.perf_counter()
        weighted_up, _probs = self.predict_proba_up(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prediction = float(weighted_up - 0.5) * 2.0   # map [0,1] → [-1,+1]
        confidence = float(max(weighted_up, 1.0 - weighted_up))
        return {
            "prediction": np.array([prediction]),
            "confidence": np.array([confidence]),
            "latency_ms": latency_ms,
            "version":    self.model_version,
        }

    # ── Dynamic weight refresh (called during retrain) ───────────────────────

    def refresh_weights(self, X_val, y_val, pnl_fn, sharpe_fn,
                        ema: float = 0.10):
        """Re-evaluate per-member Sharpe on fresh val data and EMA-update weights.
        new_weight = (1-ema)·old + ema·softmax(new_sharpe)
        """
        if not self.members or self.scaler is None:
            return
        Xs = self.scaler.transform(X_val)
        sharpes = {}
        for name, model in self.members.items():
            try:
                pred = model.predict(Xs)
                pnl  = pnl_fn(y_val, pred)
                sharpes[name] = float(sharpe_fn(pnl))
            except Exception:
                sharpes[name] = 0.0

        sh = np.array([sharpes[n] for n in MEMBER_NAMES], dtype=float)
        if np.all(sh <= 0):
            new_w = np.array([1 / len(MEMBER_NAMES)] * len(MEMBER_NAMES))
        else:
            sh = np.clip(sh, -2.0, 5.0)
            ex = np.exp(sh)
            new_w = ex / ex.sum()

        for i, name in enumerate(MEMBER_NAMES):
            old = self.weights.get(name, 1 / len(MEMBER_NAMES))
            self.weights[name] = float((1 - ema) * old + ema * new_w[i])
        # Renormalise
        total = sum(self.weights.values()) or 1.0
        for n in self.weights:
            self.weights[n] /= total
        logger.info(
            f"Ensemble weights refreshed: "
            + " ".join(f"{n}={self.weights[n]:.2f}" for n in MEMBER_NAMES)
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump({
            "scaler":  self.scaler,
            "members": self.members,
            "weights": self.weights,
            "metrics": self.member_metrics,
        }, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        data = joblib.load(path)
        inst = cls()
        inst.scaler         = data.get("scaler")
        inst.members        = data.get("members", {})
        inst.weights        = data.get("weights", {n: 1 / 3 for n in MEMBER_NAMES})
        inst.member_metrics = data.get("metrics", {})
        return inst
