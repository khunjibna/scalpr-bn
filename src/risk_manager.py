"""Risk Management — V2 Production
Position sizing, SL/TP, daily/portfolio loss guards, rolling performance,
and kill-switch conditions (handbook §8).
"""
from collections import deque
from datetime import date, datetime

import numpy as np
from loguru import logger


class RiskManager:
    def __init__(self, config: dict):
        self.config   = config
        self.risk_cfg = config.get("risk", {})

        # Daily P&L tracking
        self.daily_pnl   = 0.0
        self.daily_loss  = 0.0
        self._reset_date = date.today()

        # Portfolio peak-equity for drawdown guard
        self._peak_equity: float = 0.0

        # Kill-switch state
        self.trading_halted: bool = False
        self._halt_reason:   str  = ""

        # Rolling performance (last N trades) for kill-switch §8.2
        _window = self.risk_cfg.get("rolling_window", 50)
        self._trade_pnl_history: deque = deque(maxlen=_window)

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
        self, balance: float, entry_price: float, stop_price: float,
        win_rate: float | None = None, payoff_ratio: float | None = None,
    ) -> float:
        """
        Kelly-adjusted position sizing (handbook §7.1).
        f* = (p×b − (1−p)) / b  where p=win_rate, b=payoff_ratio
        Uses half-Kelly capped at max_position_pct.
        Falls back to fixed-fractional when stats are unavailable.
        """
        max_pct = self.risk_cfg.get("max_position_pct", 0.02)

        if win_rate is not None and payoff_ratio is not None and payoff_ratio > 0 and 0 < win_rate < 1:
            kelly_raw = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            if kelly_raw > 0:
                # Half-Kelly for safety, capped at config max
                kelly_fraction = min(kelly_raw * 0.5, max_pct)
            else:
                # Negative edge → use 25% of max until edge improves
                kelly_fraction = max_pct * 0.25
            logger.debug(
                f"Kelly: p={win_rate:.2f} b={payoff_ratio:.2f} "
                f"raw={kelly_raw:.3f} → half={kelly_fraction:.4f} (cap={max_pct:.3f})"
            )
        else:
            kelly_fraction = max_pct  # fixed-fractional fallback

        risk_amount = balance * kelly_fraction
        price_risk  = abs(entry_price - stop_price)
        if price_risk == 0:
            return 0.0
        return risk_amount / price_risk

    # ── Stop loss / Take profit ───────────────────────────────────────────────

    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """ATR-based stop loss (1.5× ATR) or percentage-based — whichever is wider."""
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
        Checks: kill-switch, daily loss, portfolio drawdown (handbook §8.2).
        """
        self._maybe_reset()

        # Kill-switch guard
        if self.trading_halted:
            return False, f"Kill-switch active: {self._halt_reason}"

        # Daily loss limit
        max_loss_pct = self.risk_cfg.get("max_daily_loss_pct", 0.05)
        if balance > 0 and (self.daily_loss / balance) >= max_loss_pct:
            return False, (
                f"Daily loss limit hit: {self.daily_loss:.2f} USDT "
                f"({self.daily_loss / balance * 100:.1f}%)"
            )

        # Portfolio drawdown guard
        if self._peak_equity > 0 and balance > 0:
            dd = (self._peak_equity - balance) / self._peak_equity
            max_dd = self.risk_cfg.get("max_portfolio_dd", 0.25)
            if dd >= max_dd:
                return False, (
                    f"Portfolio drawdown {dd:.1%} ≥ {max_dd:.0%} limit"
                )

        return True, "OK"

    # ── Kill-switch conditions ────────────────────────────────────────────────

    def check_kill_switch(
        self,
        balance: float,
        ml_drift_score: float | None = None,
        exchange_connected: bool = True,
    ) -> tuple:
        """
        Evaluate V2 kill-switch conditions (handbook §8.2).
        Returns (triggered, reason) and sets self.trading_halted if triggered.
        Conditions:
        1. Portfolio drawdown > 50%
        2. Daily loss > 5%
        3. Rolling Sharpe < 0.5 (over rolling_window trades)
        4. Feature drift KL > threshold
        5. Exchange disconnected
        """
        # Already halted — skip re-evaluation to avoid repeated CRITICAL logs
        if self.trading_halted:
            return True, self._halt_reason

        self._maybe_reset()
        triggered = []

        # 1. Portfolio drawdown > 50%
        if self._peak_equity > 0 and balance > 0:
            dd = (self._peak_equity - balance) / self._peak_equity
            if dd > 0.50:
                triggered.append(f"Portfolio DD={dd:.1%} > 50%")

        # 2. Daily loss > 5%
        if balance > 0 and (self.daily_loss / (balance + 1e-9)) > 0.05:
            triggered.append(
                f"Intraday loss={self.daily_loss / balance:.1%} > 5%"
            )

        # 3. Rolling Sharpe < 0.5
        pnl_arr = np.array(self._trade_pnl_history)
        maxlen  = self._trade_pnl_history.maxlen or 50
        min_obs = max(10, maxlen // 5)
        if len(pnl_arr) >= min_obs:
            mu  = np.mean(pnl_arr)
            sig = np.std(pnl_arr, ddof=1) + 1e-9
            rolling_sharpe = float(mu / sig * np.sqrt(252))
            if rolling_sharpe < 0.5:
                triggered.append(f"Rolling Sharpe={rolling_sharpe:.3f} < 0.5")

        # 4. Feature drift (only after enough rolling observations)
        drift_kill_thr = self.risk_cfg.get("drift_kill_threshold", 0.20)
        if ml_drift_score is not None and ml_drift_score > drift_kill_thr:
            triggered.append(f"Feature drift KL={ml_drift_score:.4f} > {drift_kill_thr}")

        # 5. Exchange disconnected
        if not exchange_connected:
            triggered.append("Exchange disconnected")

        if triggered:
            self.trading_halted = True
            self._halt_reason   = " | ".join(triggered)
            logger.critical(f"KILL-SWITCH TRIGGERED: {self._halt_reason}")
            return True, self._halt_reason

        return False, ""

    def manual_halt(self, reason: str = "Manual stop"):
        """Operator-triggered halt."""
        self.trading_halted = True
        self._halt_reason   = reason
        logger.warning(f"Manual halt: {reason}")

    def clear_halt(self):
        """Operator clears the kill-switch after investigation."""
        self.trading_halted = False
        self._halt_reason   = ""
        logger.info("Kill-switch cleared by operator")

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_trade(self, pnl: float):
        """Call after each trade closes to update daily P&L and rolling history."""
        self._maybe_reset()
        self.daily_pnl += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)
        self._trade_pnl_history.append(pnl)
        logger.info(f"Trade closed | PnL: {pnl:+.2f} | Daily: {self.daily_pnl:+.2f} USDT")

    def update_equity(self, balance: float):
        """Update peak equity for drawdown tracking. Call each cycle."""
        if balance <= 0:
            return  # ignore zero/invalid balance (API error)
        if self._peak_equity <= 0:
            # First valid reading — initialise peak so DD starts from here
            self._peak_equity = balance
            logger.debug(f"Peak equity initialised: {balance:.2f} USDT")
        elif balance > self._peak_equity:
            self._peak_equity = balance

    def get_daily_stats(self) -> dict:
        self._maybe_reset()
        return {
            "daily_pnl":       self.daily_pnl,
            "daily_loss":      self.daily_loss,
            "date":            str(self._reset_date),
            "peak_equity":     self._peak_equity,
            "trading_halted":  self.trading_halted,
            "halt_reason":     self._halt_reason,
        }

