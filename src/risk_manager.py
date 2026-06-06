"""Risk Management — position sizing, SL/TP, daily loss guard"""
from datetime import date

from loguru import logger


class RiskManager:
    def __init__(self, config: dict):
        self.config   = config
        self.risk_cfg = config.get("risk", {})

        self.daily_pnl   = 0.0
        self.daily_loss  = 0.0          # Accumulated absolute loss this day
        self._reset_date = date.today()

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _maybe_reset(self):
        today = date.today()
        if today != self._reset_date:
            self.daily_pnl  = 0.0
            self.daily_loss = 0.0
            self._reset_date = today
            logger.info("Daily P&L counters reset for new day")

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self, balance: float, entry_price: float, stop_price: float
    ) -> float:
        """
        Fixed-fractional sizing.
        Risk max_position_pct of balance; divide by per-unit price risk.
        Returns quantity in base asset.
        """
        max_pct     = self.risk_cfg.get("max_position_pct", 0.02)
        risk_amount = balance * max_pct
        price_risk  = abs(entry_price - stop_price)
        if price_risk == 0:
            return 0.0
        return risk_amount / price_risk

    # ── Stop loss / Take profit ───────────────────────────────────────────────

    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        ATR-based stop loss (1.5× ATR) or percentage-based — whichever is wider.
        side: 'LONG' | 'SHORT'
        """
        sl_pct   = self.risk_cfg.get("stop_loss_pct", 0.015)
        distance = max(atr * 1.5, entry_price * sl_pct)
        return entry_price - distance if side == "LONG" else entry_price + distance

    def calculate_take_profit(
        self, entry_price: float, stop_price: float, side: str
    ) -> float:
        """Take-profit based on R:R ratio."""
        rr     = self.risk_cfg.get("take_profit_ratio", 2.0)
        reward = abs(entry_price - stop_price) * rr
        return entry_price + reward if side == "LONG" else entry_price - reward

    # ── Trading gate ──────────────────────────────────────────────────────────

    def can_trade(self, balance: float) -> tuple:
        """
        Returns (True, 'OK') or (False, reason_string).
        """
        self._maybe_reset()
        max_loss_pct = self.risk_cfg.get("max_daily_loss_pct", 0.05)
        if balance > 0 and (self.daily_loss / balance) >= max_loss_pct:
            return False, f"Daily loss limit hit: {self.daily_loss:.2f} USDT ({self.daily_loss/balance*100:.1f}%)"
        return True, "OK"

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_trade(self, pnl: float):
        """Call after each trade closes to update daily P&L."""
        self._maybe_reset()
        self.daily_pnl += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)
        logger.info(f"Trade closed | PnL: {pnl:+.2f} | Daily: {self.daily_pnl:+.2f} USDT")

    def get_daily_stats(self) -> dict:
        self._maybe_reset()
        return {
            "daily_pnl":  self.daily_pnl,
            "daily_loss": self.daily_loss,
            "date":       str(self._reset_date),
        }
