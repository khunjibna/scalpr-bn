"""Standardized API & interface contracts (handbook §14).

This module defines the single source-of-truth for cross-model contracts:
    - ModelInterface : abstract base class (§14.1)
    - Signal         : signal output schema (§14.2)
    - ExecutionOrder : execution order schema (§14.3)

Every archetype (EnsembleModel, APV-PLN, future models) MUST emit a Signal
and conform to ModelInterface so the bot / risk / execution layers stay
loosely coupled.
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# §14.1 — Model inference interface
# ─────────────────────────────────────────────────────────────────────────────

class ModelInterface(ABC):
    """All models (sklearn ensemble, APV-PLN, etc.) implement this contract.

    Returned dict from ``predict()`` is ALWAYS:
        {
            "prediction":  float | np.ndarray,   # raw model output (e.g. expected return, class id)
            "confidence":  float | np.ndarray,   # in [0, 1]
            "latency_ms":  float,
            "version":     str,                   # model_version string
        }
    """

    config: dict
    backend: str = "cpu"
    model_version: str = "unknown"

    @abstractmethod
    def predict(self, features: np.ndarray) -> dict:
        """Run a single (or batch) inference.

        ``features`` may be (feature_dim,) or (batch, feature_dim).
        Implementations should always coerce to 2-D internally.
        """
        ...

    def get_config(self) -> dict:
        return dict(self.config or {})


# ─────────────────────────────────────────────────────────────────────────────
# §14.2 — Signal output schema
# ─────────────────────────────────────────────────────────────────────────────

Archetype = Literal[
    "trend", "mean_reversion", "scalper", "stat_arb",
    "discretionary", "market_maker", "ensemble", "apv_pln",
]


@dataclass
class Signal:
    """Standard signal emitted by any archetype / ensemble."""

    timestamp: datetime
    archetype: str
    symbol:    str

    # Prediction
    direction:  float   # -1 (SELL) / 0 (NEUTRAL) / +1 (BUY) or continuous [-1, 1]
    confidence: float   # [0, 1]

    # Model attribution
    model_name:    str
    model_version: str

    # Ensemble info
    ensemble_size:    int   = 1
    ensemble_entropy: float = 0.0  # 0=unanimous, 1=max disagreement

    # Metadata
    features_version:     str   = "2.2"
    backend:              str   = "cpu"
    inference_latency_ms: float = 0.0

    # Optional risk-management suggestions
    stop_loss:                Optional[float] = None
    take_profit:              Optional[float] = None
    position_size_suggestion: Optional[float] = None

    # Optional unique id (auto-generated if absent)
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def neutral(cls, symbol: str, archetype: str = "ensemble",
                model_name: str = "n/a", model_version: str = "n/a") -> "Signal":
        return cls(
            timestamp=datetime.now(timezone.utc),
            archetype=archetype, symbol=symbol,
            direction=0.0, confidence=0.0,
            model_name=model_name, model_version=model_version,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §14.3 — Execution order interface
# ─────────────────────────────────────────────────────────────────────────────

Side      = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT", "TWAP", "VWAP", "IOC"]
TIF       = Literal["GTC", "IOC", "FOK"]


@dataclass
class ExecutionOrder:
    """Standard order — emitted by the execution layer, consumed by exchange client."""

    symbol:     str
    side:       Side
    quantity:   float
    order_type: OrderType = "MARKET"

    limit_price:   Optional[float] = None
    time_in_force: TIF             = "GTC"

    # Traceability
    order_id:  str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    signal_id: str = ""
    archetype: str = "ensemble"

    # Risk management
    stop_loss:         Optional[float] = None
    take_profit:       Optional[float] = None
    max_position_hours: int            = 24

    # Execution constraints
    max_slippage_bps:    int   = 10    # reject if slippage > N bps
    min_fill_percentage: float = 0.80

    def to_order_message(self) -> dict:
        """Exchange-API friendly payload (Binance Futures-compatible keys)."""
        return {
            "symbol":        self.symbol,
            "side":          self.side,
            "type":          self.order_type,
            "quantity":      self.quantity,
            "price":         self.limit_price if self.order_type == "LIMIT" else None,
            "timeInForce":   self.time_in_force,
            "clientOrderId": self.order_id,
            "stopPrice":     self.stop_loss,
            "takeProfit":    self.take_profit,
        }

    def to_dict(self) -> dict:
        return asdict(self)
