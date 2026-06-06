"""Machine Learning Strategy — Random Forest Classifier"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features fed to the model (must exist in the indicator DataFrame)
FEATURE_COLS = [
    "rsi",
    "macd_diff",
    "bb_pct",
    "stoch_k",
    "stoch_d",
    "ema_fast_slow_ratio",
    "price_ema_fast_ratio",
    "price_ema_slow_ratio",
    "price_vwap_ratio",
    "vol_ratio",
    "vol_delta",
    "return_1",
    "return_3",
    "return_5",
    "atr",
    "body_ratio",
]


class MLStrategy:
    def __init__(self, config: dict):
        self.config    = config
        self.ml_cfg    = config.get("ml", {})
        self.model_path  = self.ml_cfg.get("model_path", "models/rf_model.pkl")
        self.scaler_path = self.model_path.replace(".pkl", "_scaler.pkl")
        self.model:  RandomForestClassifier | None = None
        self.scaler: StandardScaler         | None = None
        self.last_trained: datetime | None = None
        self.is_trained = False
        self._load_model()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_model(self):
        """Load a previously saved model from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model  = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info("ML model loaded from disk")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model,  self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved → {self.model_path}")

    # ── Data preparation ─────────────────────────────────────────────────────

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in FEATURE_COLS if c in df.columns]
        return df[available].copy()

    @staticmethod
    def _labels(df: pd.DataFrame, lookahead: int = 3) -> pd.Series:
        """1 if price rises over next `lookahead` candles, else 0."""
        future_ret = df["close"].shift(-lookahead) / df["close"] - 1
        return (future_ret > 0).astype(int)

    # ── Training ─────────────────────────────────────────────────────────────

    def train_model(self, df: pd.DataFrame = None) -> bool:
        """
        Train Random Forest on `df` (must already have indicator columns).
        Returns True on success.
        """
        if df is None or df.empty:
            logger.error("No data provided for training")
            return False

        min_samples = self.ml_cfg.get("min_train_samples", 200)
        feat_df = self._features(df)
        lookahead = self.ml_cfg.get("lookahead_candles", 3)
        labels  = self._labels(df, lookahead=lookahead)

        combined = pd.concat([feat_df, labels.rename("label")], axis=1).dropna()
        if len(combined) < min_samples:
            logger.warning(f"Not enough clean samples: {len(combined)} < {min_samples} — using all available")
            # ตั้ง last_trained เพื่อหยุด retrain loop (retry ใน 30 นาที)
            self.last_trained = datetime.now()
            if len(combined) < 50:
                return False

        X = combined[feat_df.columns].values
        y = combined["label"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=self.ml_cfg.get("n_estimators", 200),
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train_s, y_train)

        train_acc = accuracy_score(y_train, self.model.predict(X_train_s))
        test_acc  = accuracy_score(y_test,  self.model.predict(X_test_s))
        logger.info(f"RF trained | train_acc={train_acc:.3f} | test_acc={test_acc:.3f} | samples={len(combined)}")

        self._save_model()
        self.is_trained   = True
        self.last_trained = datetime.now()
        return True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple:
        """
        Predict direction from the latest row of `df`.
        Returns (signal, confidence) where signal ∈ {1, -1, 0}.
        """
        if not self.is_trained or self.model is None:
            return 0, 0.0

        threshold = self.ml_cfg.get("confidence_threshold", 0.58)
        feat_df   = self._features(df)

        if feat_df.empty or feat_df.isnull().any().any():
            return 0, 0.0

        latest = feat_df.iloc[-1:].values
        try:
            scaled = self.scaler.transform(latest)
            proba  = self.model.predict_proba(scaled)[0]  # [prob_down, prob_up]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0

        prob_up   = float(proba[1])
        prob_down = float(proba[0])

        if prob_up   >= threshold:
            return  1, prob_up
        if prob_down >= threshold:
            return -1, prob_down
        return 0, max(prob_up, prob_down)

    def needs_retraining(self) -> bool:
        if not self.is_trained or self.last_trained is None:
            return True
        hours_since = (datetime.now() - self.last_trained).total_seconds() / 3600
        return hours_since >= self.ml_cfg.get("retrain_hours", 24)
