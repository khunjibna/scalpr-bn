"""Machine Learning Strategy — V2 Production
Walk-forward validation, multi-seed training, comprehensive metrics,
feature drift detection, and automated retraining triggers.
"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, message=".*sklearn.utils.parallel.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*propagate the scikit-learn configuration.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Nuclear option: monkey-patch sklearn _FuncWrapper to suppress parallel warning ─
# Some sklearn 1.9+ code paths emit UserWarning from _FuncWrapper.__call__ when
# delayed() is invoked without going through sklearn.utils.parallel.Parallel
# (e.g. when joblib executes tasks across thread boundaries with n_jobs=1).
# Filter rules may not always catch them across thread contexts, so we patch
# the source to skip the warning emission entirely.
try:
    from sklearn.utils import parallel as _sk_parallel
    _orig_funcwrapper_call = _sk_parallel._FuncWrapper.__call__

    def _silent_funcwrapper_call(self, *args, **kwargs):
        from sklearn import config_context
        cfg = getattr(self, "config", None) or {}
        with config_context(**cfg):
            return self.function(*args, **kwargs)

    _sk_parallel._FuncWrapper.__call__ = _silent_funcwrapper_call
except Exception:
    pass  # keep going if sklearn internals change

from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .ensemble import EnsembleModel

# V2 Feature schema version — must match indicators.py
FEATURE_SCHEMA_VERSION = "2.2"  # 2.2 = ensemble (RF+GBM+ET)

# V2 expanded feature set
FEATURE_COLS = [
    # V2 causal returns / statistical features
    "log_return",
    "zscore_close_64",
    "ema_spread_12_26",
    "price_slope_20",
    # Momentum
    "rsi",
    "macd_diff",
    "stoch_k",
    "stoch_d",
    # Volatility
    "bb_pct",
    "atr_14",
    # Volume
    "vol_ratio",
    "vol_delta",
    # Legacy returns
    "return_1",
    "return_3",
    "return_5",
    # Ratios
    "ema_fast_slow_ratio",
    "price_ema_fast_ratio",
    "price_ema_slow_ratio",
    "price_vwap_ratio",
    # Candle structure
    "body_ratio",
]

# V2 Training seeds for multi-seed robustness (handbook §4.1)
TRAINING_SEEDS = [42, 123, 456]


# ── Metrics helpers ──────────────────────────────────────────────────────────

def _compute_sharpe(pnl_series: np.ndarray, annualise_factor: float = 252.0) -> float:
    if len(pnl_series) < 2:
        return 0.0
    mu  = np.mean(pnl_series)
    sig = np.std(pnl_series, ddof=1) + 1e-9
    return float(mu / sig * np.sqrt(annualise_factor))


def _compute_sortino(pnl_series: np.ndarray, annualise_factor: float = 252.0) -> float:
    if len(pnl_series) < 2:
        return 0.0
    mu       = np.mean(pnl_series)
    downside = pnl_series[pnl_series < 0]
    if len(downside) == 0:
        return float(mu * np.sqrt(annualise_factor) / 1e-9)  # no losses
    raw_std  = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    # floor: at least 1% of mean-absolute-loss to prevent division-by-zero explosion
    sig_down = max(raw_std, abs(np.mean(downside)) * 0.01) + 1e-9
    return float(np.clip(mu / sig_down * np.sqrt(annualise_factor), -100.0, 100.0))


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd   = (peak - equity_curve) / (np.abs(peak) + 1e-9)
    return float(np.max(dd))


def _compute_calmar(pnl_series: np.ndarray, annualise_factor: float = 252.0) -> float:
    equity = np.cumsum(pnl_series) + 1.0
    mdd    = _compute_max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(np.mean(pnl_series) * annualise_factor / mdd)


def _compute_profit_factor(pnl_series: np.ndarray) -> float:
    wins   = pnl_series[pnl_series > 0].sum()
    losses = np.abs(pnl_series[pnl_series < 0].sum())
    if losses == 0:
        return float(wins) if wins > 0 else 0.0
    return float(wins / losses)


def _compute_win_rate(pnl_series: np.ndarray) -> float:
    if len(pnl_series) == 0:
        return 0.0
    return float(np.sum(pnl_series > 0) / len(pnl_series))


def _compute_payoff_ratio(pnl_series: np.ndarray) -> float:
    wins   = pnl_series[pnl_series > 0]
    losses = np.abs(pnl_series[pnl_series < 0])
    avg_win  = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-9
    return float(avg_win / avg_loss) if avg_loss > 0 else 0.0


def _simulate_pnl(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Simulated per-bar P&L for metric calculation.
    Uses ±0.001 (0.1% per bar) so the equity curve stays near 1.0
    and MaxDrawdown / Sortino remain in realistic ranges.
    """
    correct = (y_pred == y_true).astype(float)
    return (correct - (1.0 - correct)) * 0.001


# ── Walk-forward split generator ─────────────────────────────────────────────

def _walk_forward_splits(n: int, n_walks: int = 5,
                         train_pct: float = 0.65,
                         val_pct: float = 0.15) -> list:
    stride  = max(1, n // (n_walks + 1))
    splits  = []
    for i in range(n_walks):
        start     = i * stride
        train_end = start + int(train_pct * n)
        val_end   = train_end + int(val_pct * n)
        test_end  = min(n, val_end + int((1 - train_pct - val_pct) * n))
        if test_end <= val_end or train_end >= n:
            continue
        splits.append({
            "walk":  i,
            "train": slice(start, train_end),
            "val":   slice(train_end, val_end),
            "test":  slice(val_end, test_end),
        })
    return splits


# ── Data quality validation ──────────────────────────────────────────────────

def _validate_data(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
    if np.any(~np.isfinite(X_train)):
        raise ValueError("NaN/Inf in training features")
    if len(X_val) > 0 and np.any(~np.isfinite(X_val)):
        raise ValueError("NaN/Inf in validation features")
    if len(y_train) > 0:
        pos = np.mean(y_train)
        neg = 1 - pos
        imbalance = max(pos, neg) / (min(pos, neg) + 1e-9)
        if imbalance > 5.0:
            logger.warning(f"Class imbalance {imbalance:.1f}× — consider resampling")


# ── Feature drift detector ───────────────────────────────────────────────────

class _DriftDetector:
    """KL-divergence feature drift detector (handbook §12.2)."""

    def __init__(self):
        self.train_mean: np.ndarray | None = None
        self.train_std:  np.ndarray | None = None
        self._history: deque = deque(maxlen=500)

    def fit(self, X_train: np.ndarray):
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std  = np.std(X_train,  axis=0) + 1e-8
        # Reset history so post-retrain scores are measured against the new baseline
        self._history.clear()

    def score(self, X_batch: np.ndarray) -> float:
        if self.train_mean is None or len(X_batch) == 0:
            return 0.0
        live_mean = np.mean(X_batch, axis=0)
        live_std  = np.std(X_batch,  axis=0) + 1e-8
        kl = 0.5 * (
            (live_std ** 2 + (live_mean - self.train_mean) ** 2)
            / (self.train_std ** 2)
            - 1
            + 2 * np.log(self.train_std / live_std)
        )
        s = float(np.clip(np.mean(kl), 0, 1e6))
        self._history.append(s)
        return s

    def rolling_score(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def is_drifting(self, X_batch: np.ndarray, threshold: float = 0.10,
                    min_obs: int = 20) -> bool:
        """Only flag drift after enough observations to reduce cold-start noise."""
        self.score(X_batch)
        if len(self._history) < min_obs:
            return False  # not enough data yet — skip
        return self.rolling_score() > threshold




# ── MLStrategy ───────────────────────────────────────────────────────────────

class MLStrategy:
    def __init__(self, config: dict):
        self.config    = config
        self.ml_cfg    = config.get("ml", {})
        self.model_path  = self.ml_cfg.get("model_path", "models/rf_model.pkl")
        self.scaler_path = self.model_path.replace(".pkl", "_scaler.pkl")
        self.metrics_path = self.model_path.replace(".pkl", "_metrics.pkl")
        # Ensemble lives next to the legacy model file with a distinct suffix
        self.ensemble_path = self.model_path.replace(".pkl", "_ensemble.pkl")

        self.model:    RandomForestClassifier | None = None   # legacy fallback
        self.scaler:   StandardScaler         | None = None   # legacy fallback
        self.ensemble: EnsembleModel          | None = None
        self.drift_detector = _DriftDetector()

        self.last_trained: datetime | None = None
        self.is_trained = False

        # Rolling trade P&L history for performance-based retrain triggers
        self._trade_pnl_history: deque = deque(
            maxlen=self.ml_cfg.get("rolling_window", 100)
        )
        self._last_validation_metrics: dict = {}

        self._load_model()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_model(self):
        # Prefer ensemble model if present
        if os.path.exists(self.ensemble_path):
            try:
                ens = EnsembleModel.load(self.ensemble_path)
                saved_n = getattr(ens.scaler, "n_features_in_", None)
                if saved_n is not None and saved_n != len(FEATURE_COLS):
                    logger.warning(
                        f"Stale ensemble discarded: saved={saved_n} features, "
                        f"current schema={len(FEATURE_COLS)} — will retrain"
                    )
                    return
                self.ensemble = ens
                self.is_trained = True
                if os.path.exists(self.metrics_path):
                    self._last_validation_metrics = joblib.load(self.metrics_path)
                logger.info(
                    f"Ensemble loaded | schema={FEATURE_SCHEMA_VERSION} | "
                    f"features={saved_n} | weights={ens.weights}"
                )
                return
            except Exception as e:
                logger.warning(f"Could not load ensemble: {e}")

        # Legacy single-model fallback
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                model  = joblib.load(self.model_path)
                scaler = joblib.load(self.scaler_path)

                # Validate feature count against current schema
                saved_n = getattr(scaler, 'n_features_in_', None)
                if saved_n is not None and saved_n != len(FEATURE_COLS):
                    logger.warning(
                        f"Stale model discarded: saved={saved_n} features, "
                        f"current schema={len(FEATURE_COLS)} — will retrain"
                    )
                    return  # leave model/scaler as None → triggers retrain

                self.model  = model
                self.scaler = scaler
                self.is_trained = True
                if os.path.exists(self.metrics_path):
                    self._last_validation_metrics = joblib.load(self.metrics_path)
                logger.info(
                    f"Legacy single-model loaded | schema={FEATURE_SCHEMA_VERSION} | "
                    f"features={saved_n} | will upgrade to ensemble on next retrain"
                )
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if self.ensemble is not None:
            self.ensemble.save(self.ensemble_path)
            joblib.dump(self._last_validation_metrics, self.metrics_path)
            # Remove stale legacy single-model files to avoid confusion
            for p in (self.model_path, self.scaler_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            logger.info(f"Ensemble saved → {self.ensemble_path}")
        else:
            joblib.dump(self.model,  self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self._last_validation_metrics, self.metrics_path)
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

    @staticmethod
    def _denoise_mask(df: pd.DataFrame, lookahead: int,
                      min_move_atr: float) -> pd.Series:
        """Return boolean mask of samples whose |future_ret| ≥ min_move_atr × ATR%.
        Used to drop ambiguous "noise" bars from training so the ML model
        only learns directional moves larger than typical bar volatility.
        min_move_atr=0 disables the filter.
        """
        if min_move_atr <= 0 or "atr_14" not in df.columns:
            return pd.Series(True, index=df.index)
        future_ret = (df["close"].shift(-lookahead) / df["close"] - 1).abs()
        atr_pct    = df["atr_14"] / df["close"]
        threshold  = atr_pct * min_move_atr
        return future_ret >= threshold

    # ── Single-seed training ──────────────────────────────────────────────────

    # ── Optuna hyperparameter search (handbook §4.2) ────────────────────────

    def _optuna_hp_search(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          n_trials: int = 50) -> dict:
        """Optuna hyperparameter search for RandomForest (handbook §4.2).
        Returns best params dict (empty dict on error / optuna not installed)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed — skipping HP search (pip install optuna)")
            return {}

        from sklearn import config_context, get_config
        _sk_cfg = get_config()   # capture current thread's sklearn config

        def _objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
                "max_depth":        trial.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
                "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            with config_context(**_sk_cfg):   # propagate sklearn config → suppress warning
                sc  = StandardScaler()
                Xtr = sc.fit_transform(X_train)
                Xva = sc.transform(X_val)
                clf = RandomForestClassifier(**params, n_jobs=1, random_state=42)
                clf.fit(Xtr, y_train)
            pnl = _simulate_pnl(y_val, clf.predict(Xva))
            return _compute_sharpe(pnl)

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        logger.info(f"Optuna HP search | best={best} | sharpe={study.best_value:.3f}")
        return best

    # ── Regime-stratified evaluation (handbook §5.2) ─────────────────────

    @staticmethod
    def _regime_labels(X: np.ndarray) -> np.ndarray:
        """Label bars as 0=ranging / 1=trending via ATR relative to median."""
        try:
            atr_idx = FEATURE_COLS.index("atr_14")
            vol = np.abs(X[:, atr_idx])
        except (ValueError, IndexError):
            vol = np.std(X, axis=1)
        return (vol > np.median(vol)).astype(int)

    def _regime_eval(self, X_val: np.ndarray, y_val: np.ndarray,
                     model, scaler) -> dict:
        """Evaluate model Sharpe / win-rate per market regime (handbook §5.2)."""
        Xva_s  = scaler.transform(X_val)
        y_pred = model.predict(Xva_s)
        pnl    = _simulate_pnl(y_val, y_pred)
        regimes = self._regime_labels(X_val)
        result: dict = {}
        for regime_id, name in [(0, "ranging"), (1, "trending")]:
            mask = regimes == regime_id
            if mask.sum() < 10:
                continue
            r_pnl = pnl[mask]
            result[name] = {
                "n":        int(mask.sum()),
                "sharpe":   round(_compute_sharpe(r_pnl), 3),
                "win_rate": round(_compute_win_rate(r_pnl), 3),
            }
        return result

    # ── Single-seed training ────────────────────────────────────────────────

    def _train_single_seed(self, X_train, y_train, X_val, y_val, seed: int,
                           hp_overrides: dict | None = None) -> dict:
        np.random.seed(seed)
        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(X_train)
        Xva_s  = scaler.transform(X_val)

        params = {
            "n_estimators":     self.ml_cfg.get("n_estimators", 200),
            "max_depth":        10,
            "min_samples_leaf": 10,
            "max_features":     "sqrt",
        }
        if hp_overrides:
            params.update(hp_overrides)

        model = RandomForestClassifier(
            **params,
            n_jobs=1,   # single-thread: avoids sklearn parallel config warning
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(Xtr_s, y_train)

        val_pred = model.predict(Xva_s)
        val_acc  = float(accuracy_score(y_val, val_pred))
        pnl      = _simulate_pnl(y_val, val_pred)

        return {
            "seed":              seed,
            "model":             model,
            "scaler":            scaler,
            "val_acc":           val_acc,
            "val_sharpe":        _compute_sharpe(pnl),
            "val_sortino":       _compute_sortino(pnl),
            "val_calmar":        _compute_calmar(pnl),
            "val_max_dd":        _compute_max_drawdown(np.cumsum(pnl) + 1.0),
            "val_profit_factor": _compute_profit_factor(pnl),
            "val_win_rate":      _compute_win_rate(pnl),
            "val_payoff":        _compute_payoff_ratio(pnl),
        }

    # ── Validation gates ──────────────────────────────────────────────────────

    def _check_validation_gates(self, m: dict) -> tuple:
        """V2 gates (handbook §1.3). Returns (passed, failures_list)."""
        checks = [
            ("Sharpe",        m.get("val_sharpe",        0.0), ">=", 1.2),
            ("Accuracy",      m.get("val_acc",            0.0), ">=", 0.55),
            ("ProfitFactor",  m.get("val_profit_factor",  0.0), ">=", 1.5),
            ("MaxDrawdown",   m.get("val_max_dd",         1.0), "<=", 0.20),
            ("Calmar",        m.get("val_calmar",         0.0), ">=", 1.0),
            ("Sortino",       m.get("val_sortino",        0.0), ">=", 1.5),
            ("WinRate",       m.get("val_win_rate",       0.0), ">=", 0.45),
            ("PayoffRatio",   m.get("val_payoff",         0.0), ">=", 1.0),
        ]
        failures = [
            f"{name}={val:.3f} {'<' if op == '>=' else '>'} {thr}"
            for name, val, op, thr in checks
            if (op == ">=" and val < thr) or (op == "<=" and val > thr)
        ]
        return len(failures) == 0, failures

    # ── Walk-forward Sharpe variance ─────────────────────────────────────────

    def _wf_sharpe_variance(self, X: np.ndarray, y: np.ndarray,
                             n_walks: int = 5) -> float:
        sharpes = []
        for sp in _walk_forward_splits(len(X), n_walks=n_walks):
            Xtr, ytr = X[sp["train"]], y[sp["train"]]
            Xte, yte = X[sp["test"]],  y[sp["test"]]
            if len(Xtr) < 50 or len(Xte) < 20:
                continue
            res = self._train_single_seed(Xtr, ytr, Xte, yte, seed=42)
            sharpes.append(res["val_sharpe"])
        if len(sharpes) < 2:
            return 0.0
        mean_s = np.mean(sharpes)
        return float(np.std(sharpes) / (abs(mean_s) + 1e-9))

    # ── Main training entry-point ─────────────────────────────────────────────

    def train_model(self, df: pd.DataFrame = None) -> bool:
        """
        V2 pipeline (handbook §4):
        1. Data quality validation
        2. Multi-seed training (seeds 42, 123, 456)
        3. Best-seed selection by validation Sharpe
        4. Walk-forward Sharpe variance check
        5. V2 validation gate check (8 criteria)
        6. Fit drift detector on training data
        Returns True on success.
        """
        if df is None or df.empty:
            logger.error("No data provided for training")
            return False

        min_samples = self.ml_cfg.get("min_train_samples", 200)
        lookahead   = self.ml_cfg.get("lookahead_candles", 3)
        min_move    = float(self.ml_cfg.get("min_move_atr", 0.0))

        feat_df  = self._features(df)
        labels   = self._labels(df, lookahead=lookahead)
        keep     = self._denoise_mask(df, lookahead=lookahead, min_move_atr=min_move)
        combined = pd.concat(
            [feat_df, labels.rename("label"), keep.rename("_keep")], axis=1
        ).dropna()
        n_before = len(combined)
        combined = combined[combined["_keep"]].drop(columns=["_keep"])
        if min_move > 0 and n_before:
            kept_pct = len(combined) / n_before * 100
            logger.info(
                f"Label denoise | min_move={min_move:.2f}×ATR | "
                f"kept {len(combined)}/{n_before} ({kept_pct:.1f}%)"
            )

        if len(combined) < min_samples:
            logger.warning(f"Samples: {len(combined)} < {min_samples} — using all available")
            self.last_trained = datetime.now()
            if len(combined) < 50:
                return False

        X = combined[feat_df.columns].values
        y = combined["label"].values

        # Chronological 70/15/15 split
        n       = len(X)
        tr_end  = int(n * 0.70)
        val_end = int(n * 0.85)
        X_train, y_train = X[:tr_end],          y[:tr_end]
        X_val,   y_val   = X[tr_end:val_end],   y[tr_end:val_end]
        X_test,  y_test  = X[val_end:],         y[val_end:]

        if len(X_val) < 10:
            X_val, y_val = X_test, y_test  # Fallback for tiny datasets

        # ── Data quality check ────────────────────────────────────────────────
        try:
            _validate_data(X_train, y_train, X_val, y_val)
        except ValueError as e:
            logger.error(f"Data validation failed: {e}")
            return False

        # ── Optuna HP search (§4.2) ──────────────────────────────────────
        n_trials = self.ml_cfg.get("optuna_trials", 50)
        best_hp: dict = {}
        if n_trials > 0:
            best_hp = self._optuna_hp_search(X_train, y_train, X_val, y_val,
                                             n_trials=n_trials)

        # ── Multi-seed training ──────────────────────────────────────────
        seed_results = {}
        for seed in TRAINING_SEEDS:
            try:
                res = self._train_single_seed(
                    X_train, y_train, X_val, y_val, seed, hp_overrides=best_hp
                )
                seed_results[seed] = res
                logger.info(
                    f"Seed {seed} | acc={res['val_acc']:.3f} | "
                    f"sharpe={res['val_sharpe']:.3f} | "
                    f"sortino={res['val_sortino']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Seed {seed} failed: {e}")

        if not seed_results:
            logger.error("All seeds failed — aborting training")
            return False

        best_seed = max(seed_results, key=lambda s: seed_results[s]["val_sharpe"])
        best      = seed_results[best_seed]
        mean_sharpe = np.mean([r["val_sharpe"] for r in seed_results.values()])
        std_sharpe  = np.std( [r["val_sharpe"] for r in seed_results.values()])
        logger.info(
            f"Multi-seed done | best={best_seed} | "
            f"mean_sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}"
        )

        # ── Walk-forward variance check ───────────────────────────────────────
        wf_var = self._wf_sharpe_variance(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val]),
        )
        if wf_var > 0.30:
            logger.warning(
                f"WF Sharpe variance={wf_var:.2%} > 30% — regime-sensitive model"
            )

        # ── V2 validation gates ───────────────────────────────────────────────
        passed, failures = self._check_validation_gates(best)
        if not passed:
            if not self.is_trained:
                # No existing model — deploy anyway so the bot can start trading,
                # but flag as low-quality; next retrain cycle will try to improve.
                logger.warning(
                    f"V2 gates FAILED (first train): {failures} — deploying cautiously"
                )
            else:
                # Already have a working model — keep the old one, skip this retrain
                logger.warning(
                    f"V2 gates FAILED: {failures} — keeping previous model, will retry"
                )
                self._last_validation_metrics.update({
                    "gates_passed":   False,
                    "gate_failures":  failures,
                    "wf_sharpe_variance": wf_var,
                })
                return False
        else:
            logger.success(
                f"V2 gates PASSED | sharpe={best['val_sharpe']:.3f} | "
                f"acc={best['val_acc']:.3f} | calmar={best['val_calmar']:.3f}"
            )
        # ── Regime-stratified eval (§5.2) ────────────────────────────────────
        regime_metrics = self._regime_eval(X_val, y_val, best["model"], best["scaler"])
        for regime, rm in regime_metrics.items():
            logger.info(
                f"Regime [{regime}] n={rm['n']} | "
                f"sharpe={rm['sharpe']:.3f} | win_rate={rm['win_rate']:.3f}"
            )
        # ── Build ensemble (handbook §9) ─────────────────────────────────────
        ensemble = EnsembleModel()
        try:
            ens_metrics = ensemble.fit(
                X_train, y_train, X_val, y_val,
                seed=best_seed, hp_overrides=best_hp,
                pnl_fn=_simulate_pnl, sharpe_fn=_compute_sharpe,
            )
            logger.success(
                "Ensemble fitted | "
                + " | ".join(
                    f"{n}=sharpe {m['sharpe']:.2f}" for n, m in ens_metrics.items()
                )
            )
        except Exception as e:
            logger.error(f"Ensemble training failed, falling back to single model: {e}")
            ensemble = None

        # ── Fit drift detector ────────────────────────────────────────────────
        self.drift_detector.fit(X_train)

        # ── Commit model ──────────────────────────────────────────────────────
        if ensemble is not None and ensemble.members:
            self.ensemble = ensemble
            self.model    = None   # ensemble supersedes legacy single model
            self.scaler   = None
        else:
            self.model    = best["model"]
            self.scaler   = best["scaler"]
            self.ensemble = None

        self._last_validation_metrics = {
            k: v for k, v in best.items() if k not in ("model", "scaler")
        }
        self._last_validation_metrics["wf_sharpe_variance"] = wf_var
        self._last_validation_metrics["gates_passed"]       = passed
        self._last_validation_metrics["gate_failures"]      = failures
        if self.ensemble is not None:
            self._last_validation_metrics["ensemble_weights"] = dict(self.ensemble.weights)
            self._last_validation_metrics["ensemble_members"] = self.ensemble.member_metrics

        # Test-set report (use ensemble if available, else legacy model)
        if len(X_test) >= 10:
            if self.ensemble is not None:
                Xte_s   = self.ensemble.scaler.transform(X_test)
                # Weighted hard-label vote
                probs_up = np.zeros(len(X_test))
                for n, mdl in self.ensemble.members.items():
                    probs_up += self.ensemble.weights.get(n, 0.0) * mdl.predict_proba(Xte_s)[:, 1]
                te_pred = (probs_up >= 0.5).astype(int)
            else:
                Xte_s   = self.scaler.transform(X_test)
                te_pred = self.model.predict(Xte_s)
            te_pnl    = _simulate_pnl(y_test, te_pred)
            logger.info(
                f"Test-set | acc={accuracy_score(y_test, te_pred):.3f} | "
                f"sharpe={_compute_sharpe(te_pnl):.3f} | "
                f"win_rate={_compute_win_rate(te_pnl):.3f}"
            )

        self._save_model()
        self.is_trained   = True
        self.last_trained = datetime.now()
        return True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple:
        """
        Predict direction from the latest row of `df`.
        Returns (signal, confidence, size_scalar) where:
            signal      ∈ {1, -1, 0}
            confidence  ∈ [0, 1]
            size_scalar ∈ [0, 1] — ensemble agreement multiplier (1.0 for legacy model)
        Runs feature drift check on the last 100 rows.
        """
        if not self.is_trained or (self.ensemble is None and self.model is None):
            return 0, 0.0, 0.0

        threshold = self.ml_cfg.get("confidence_threshold", 0.58)
        min_agree = self.ml_cfg.get("ensemble_min_agree", 2)
        feat_df   = self._features(df)

        if feat_df.empty or feat_df.isnull().any().any():
            return 0, 0.0, 0.0

        # Drift check on recent window
        drift_threshold = self.ml_cfg.get("drift_threshold", 0.15)
        recent = feat_df.tail(100).values
        if self.drift_detector.is_drifting(recent, threshold=drift_threshold, min_obs=20):
            drift_score = self.drift_detector.rolling_score()
            logger.warning(
                f"Feature drift KL={drift_score:.4f} > {drift_threshold} "
                f"— predictions may be unreliable"
            )

        latest = feat_df.iloc[-1:].values

        # ── Ensemble path (preferred) ────────────────────────────────────────
        if self.ensemble is not None:
            try:
                signal, conf, agreement, size_scalar, probs = self.ensemble.predict_consensus(
                    latest, conf_threshold=threshold, min_agree=min_agree,
                )
                logger.debug(
                    f"Ensemble vote | probs={ {k: round(v, 2) for k, v in probs.items()} } "
                    f"| conf={conf:.2f} | agree={agreement:.2f} | size×{size_scalar:.2f}"
                )
                return signal, conf, size_scalar
            except Exception as e:
                logger.error(f"Ensemble prediction error: {e}")
                return 0, 0.0, 0.0

        # ── Legacy single-model fallback ─────────────────────────────────────
        try:
            scaled = self.scaler.transform(latest)
            proba  = self.model.predict_proba(scaled)[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0, 0.0

        prob_up   = float(proba[1])
        prob_down = float(proba[0])

        if prob_up   >= threshold:
            return  1, prob_up,   1.0
        if prob_down >= threshold:
            return -1, prob_down, 1.0
        return 0, max(prob_up, prob_down), 0.0

    # ── Rolling performance tracking ──────────────────────────────────────────

    def record_trade_result(self, pnl: float):
        """Call after each trade closes to feed rolling metrics."""
        self._trade_pnl_history.append(pnl)

    # ── §14.2 standardised Signal emission ────────────────────────────────────

    def emit_signal(self, df: pd.DataFrame, symbol: str) -> "object":
        """Return a §14.2-compliant Signal from the latest row.
        Always returns a Signal (neutral when not trained / no consensus).
        Imported lazily so we don't introduce a hard import cycle.
        """
        from datetime import datetime, timezone
        from .interfaces import Signal

        sig, conf, size_scalar = self.predict(df) if self.is_trained else (0, 0.0, 0.0)
        backend = "ensemble" if self.ensemble is not None else (
            "sklearn_rf" if self.model is not None else "none"
        )
        version = "ensemble_v1" if self.ensemble is not None else FEATURE_SCHEMA_VERSION
        return Signal(
            timestamp=datetime.now(timezone.utc),
            archetype="ensemble" if self.ensemble is not None else "scalper",
            symbol=symbol,
            direction=float(sig),
            confidence=float(conf),
            model_name="MLStrategy",
            model_version=version,
            ensemble_size=len(self.ensemble.members) if self.ensemble else 1,
            ensemble_entropy=float(1.0 - size_scalar),
            features_version=FEATURE_SCHEMA_VERSION,
            backend=backend,
            inference_latency_ms=0.0,
            position_size_suggestion=float(size_scalar),
        )

    def get_rolling_metrics(self) -> dict:
        pnl_arr = np.array(self._trade_pnl_history)
        if len(pnl_arr) < 5:
            return {"rolling_sharpe": None, "rolling_win_rate": None,
                    "drift_score": None}
        return {
            "rolling_sharpe":        _compute_sharpe(pnl_arr),
            "rolling_win_rate":      _compute_win_rate(pnl_arr),
            "rolling_payoff_ratio":  _compute_payoff_ratio(pnl_arr),
            "drift_score":           self.drift_detector.rolling_score(),
        }

    # ── Retraining triggers ───────────────────────────────────────────────────

    def needs_retraining(self) -> bool:
        """
        V2 triggers (handbook §12.3):
        1. Not trained / first run
        2. Scheduled: retrain_hours exceeded
        3. Rolling Sharpe < 0.5
        4. Rolling win-rate < 40%
        5. Feature drift KL > 0.10
        """
        if not self.is_trained or self.last_trained is None:
            return True

        # Gates failed last time → retry sooner (1h instead of retrain_hours)
        if not self._last_validation_metrics.get("gates_passed", True):
            hours_since = (datetime.now() - self.last_trained).total_seconds() / 3600
            if hours_since >= 1.0:
                logger.info("Retrain: previous gates failed, retrying with fresh data")
                return True

        hours_since = (datetime.now() - self.last_trained).total_seconds() / 3600
        if hours_since >= self.ml_cfg.get("retrain_hours", 24):
            logger.info(f"Scheduled retrain: {hours_since:.1f}h elapsed")
            return True

        metrics  = self.get_rolling_metrics()
        min_obs  = max(20, self.ml_cfg.get("rolling_window", 100) // 5)

        if len(self._trade_pnl_history) >= min_obs:
            rs = metrics["rolling_sharpe"]
            wr = metrics["rolling_win_rate"]
            if rs is not None and rs < 0.5:
                logger.warning(f"Retrain: rolling Sharpe={rs:.3f} < 0.5")
                return True
            if wr is not None and wr < 0.40:
                logger.warning(f"Retrain: rolling win_rate={wr:.1%} < 40%")
                return True

        ds = metrics["drift_score"]
        if ds is not None and ds > 0.10:
            logger.warning(f"Retrain: feature drift KL={ds:.4f} > 0.10")
            return True

        return False

    def get_last_validation_metrics(self) -> dict:
        return self._last_validation_metrics.copy()

