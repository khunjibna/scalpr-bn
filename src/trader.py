"""Core Trading Bot"""
import os
import threading
import time
from datetime import datetime

from loguru import logger

from .binance_client import BinanceClient
from .indicators import calculate_indicators, get_signal_from_indicators
from .ml_strategy import MLStrategy
from .risk_manager import RiskManager


class TradingBot:
    def __init__(self, config: dict, client: BinanceClient = None):
        self.config      = config
        self.trading_cfg = config.get("trading", {})
        self.bot_cfg     = config.get("bot", {})
        self.risk_cfg    = config.get("risk", {})

        self.symbol      = self.trading_cfg.get("symbol",      "BTCUSDT")
        self.timeframe   = self.trading_cfg.get("timeframe",   "15m")
        self.leverage    = self.trading_cfg.get("leverage",    5)
        self.margin_type = self.trading_cfg.get("margin_type", "ISOLATED")
        self.max_pos     = self.risk_cfg.get("max_open_positions", 1)

        if client is not None:
            self.client = client
        else:
            use_testnet = self.trading_cfg.get("testnet", False) or os.getenv("USE_TESTNET", "false").lower() == "true"
            self.client = BinanceClient(testnet=use_testnet)
        self.strategy = MLStrategy(config)
        self.risk     = RiskManager(config)

        # Shared state (guarded by _lock)
        self._lock          = threading.Lock()
        self._running       = False
        self.status         = "stopped"
        self.current_signal = 0
        self.ml_confidence  = 0.0
        self.ind_signal     = 0
        self.trade_history: list[dict] = []
        self.last_update: datetime | None = None
        self.error_message  = ""

        self.step_size = 0.001
        self._position_open_time: datetime | None = None  # for max_hold_candles
        # Software SL/TP tracker: {symbol: {"sl": float, "tp": float, "side": str, "qty": float}}
        self._sltp: dict = {}
        self._use_exchange_orders = not self.trading_cfg.get("testnet", False) and \
            os.getenv("USE_TESTNET", "false").lower() != "true"
        self._setup_symbol()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_symbol(self):
        try:
            self.client.set_margin_type(self.symbol, self.margin_type)
            self.client.set_leverage(self.symbol, self.leverage)
            self.step_size = self.client.get_step_size(self.symbol)
            logger.info(f"Symbol ready: {self.symbol} | step={self.step_size}")
        except Exception as e:
            logger.error(f"Symbol setup error: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self.status   = "running"
        interval      = self.bot_cfg.get("loop_interval_seconds", 60)
        logger.info(f"Bot started | {self.symbol} {self.timeframe} {self.leverage}x | loop={interval}s")

        while self._running:
            try:
                self.run_cycle()
            except Exception as e:
                with self._lock:
                    self.error_message = str(e)
                logger.exception(f"Unhandled error in trading cycle: {e}")
            time.sleep(interval)

        self.status = "stopped"
        logger.info("Bot stopped")

    def stop(self):
        self._running = False
        self.status   = "stopping"
        logger.info("Bot stop requested")

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        with self._lock:
            self.last_update = datetime.now()

            limit = self.bot_cfg.get("klines_limit", 500)
            df = self.client.get_klines(self.symbol, self.timeframe, limit)
            if df.empty:
                logger.warning("No klines received — skipping cycle")
                return

            df = calculate_indicators(df, self.config)
            if df.empty:
                logger.warning("Indicator DataFrame empty after calculation")
                return

            # Retrain if due (run in background thread to avoid blocking the cycle)
            if self.strategy.needs_retraining():
                logger.info("Retraining ML model …")
                t = threading.Thread(
                    target=self.strategy.train_model,
                    args=(df.copy(),),
                    daemon=True,
                )
                t.start()

            # Signals
            self.ind_signal           = get_signal_from_indicators(df, self.config)
            ml_signal, self.ml_confidence = self.strategy.predict(df)

            # Both indicators and ML must agree
            self.current_signal = (
                ml_signal if (self.ind_signal == ml_signal and ml_signal != 0) else 0
            )

            price = df["close"].iloc[-1]
            logger.info(
                f"Cycle | price={price:.2f} | ind={self.ind_signal} | "
                f"ml={ml_signal}({self.ml_confidence:.2f}) | signal={self.current_signal}"
            )

            self._manage_positions(df)

            if self.current_signal != 0:
                self._process_signal(self.current_signal, df)

            self.error_message = ""

    # ── Position management ───────────────────────────────────────────────────

    def _manage_positions(self, df):
        """Log open positions + enforce software SL/TP + max_hold_candles."""
        positions = self.client.get_positions(self.symbol)
        max_hold  = self.config.get("strategy", {}).get("max_hold_candles", 0)
        price     = float(df["close"].iloc[-1])

        if not positions:
            # Position closed (by exchange SL/TP or manually) — clear tracker
            if self.symbol in self._sltp:
                del self._sltp[self.symbol]
                self._position_open_time = None
            return

        for pos in positions:
            pnl = float(pos.get("unrealizedProfit", 0))
            amt = float(pos["positionAmt"])
            logger.info(
                f"Position | amt={amt} "
                f"entry={float(pos['entryPrice']):.4f} | uPnL={pnl:+.2f}"
            )

            # ── Software SL/TP check ─────────────────────────────────────────
            sltp = self._sltp.get(self.symbol)
            if sltp:
                sl, tp, side = sltp["sl"], sltp["tp"], sltp["side"]
                hit = None
                if side == "LONG":
                    if price <= sl: hit = "STOP LOSS"
                    elif price >= tp: hit = "TAKE PROFIT"
                else:  # SHORT
                    if price >= sl: hit = "STOP LOSS"
                    elif price <= tp: hit = "TAKE PROFIT"

                if hit:
                    logger.info(f"{hit} triggered | {side} @ {price:.4f} (sl={sl:.4f} tp={tp:.4f})")
                    close_side = "SELL" if amt > 0 else "BUY"
                    self.client.close_position(self.symbol, close_side, abs(amt))
                    self.risk.record_trade(pnl)
                    del self._sltp[self.symbol]
                    self._position_open_time = None
                    # Mark latest trade as closed
                    for t in reversed(self.trade_history):
                        if t["symbol"] == self.symbol and t["status"] == "OPEN":
                            t["status"] = hit
                            t["pnl"]    = pnl
                            break
                    return

            # ── Time-stop (max_hold_candles) ─────────────────────────────────
            if max_hold > 0 and self._position_open_time is not None:
                seconds_held = (datetime.now() - self._position_open_time).total_seconds()
                tf_seconds   = _timeframe_to_seconds(self.timeframe)
                candles_held = seconds_held / tf_seconds
                if candles_held >= max_hold:
                    logger.warning(
                        f"Time-stop: held {candles_held:.1f} candles ≥ {max_hold} — force closing"
                    )
                    close_side = "SELL" if amt > 0 else "BUY"
                    self.client.close_position(self.symbol, close_side, abs(amt))
                    self.risk.record_trade(pnl)
                    self._sltp.pop(self.symbol, None)
                    self._position_open_time = None

    # ── Signal execution ──────────────────────────────────────────────────────

    def _process_signal(self, signal: int, df):
        """Evaluate signal and open a new position if all conditions pass."""
        positions = self.client.get_positions(self.symbol)

        # Max positions guard
        if len(positions) >= self.max_pos:
            logger.debug("Max positions reached — skip")
            return

        # Already in same-direction position guard
        for pos in positions:
            amt = float(pos["positionAmt"])
            if signal == 1 and amt > 0:
                return
            if signal == -1 and amt < 0:
                return

        # Balance & risk gate
        balance = self.client.get_futures_balance()
        ok, reason = self.risk.can_trade(balance)
        if not ok:
            logger.warning(f"Trade blocked: {reason}")
            return
        if balance < 5:
            logger.warning(f"Balance too low: {balance:.2f} USDT (minimum $5 required)")
            return

        # Calculate trade parameters
        price = float(df["close"].iloc[-1])
        atr   = float(df["atr"].iloc[-1])
        side  = "LONG" if signal == 1 else "SHORT"

        sl  = self.risk.calculate_stop_loss(price, side, atr)
        tp  = self.risk.calculate_take_profit(price, sl, side)

        # qty = risk_amount / sl_distance  (leverage is handled by the exchange)
        # DO NOT multiply by leverage — that amplifies incorrectly
        qty = self.risk.calculate_position_size(balance, price, sl)

        # Hard cap: margin used per trade ≤ 20% of available balance
        margin_required = qty * price / self.leverage
        max_margin      = balance * 0.20
        if margin_required > max_margin:
            qty = (max_margin * self.leverage) / price
            logger.info(f"Qty capped by margin: margin_cap={max_margin:.2f} USDT → qty={qty:.6f}")

        qty = self.client.round_quantity(qty, self.step_size)

        # Binance Futures minimum notional = $5
        if qty <= 0 or qty * price < 5:
            logger.warning(f"Qty too small: {qty} ({qty * price:.2f} USDT) — skip")
            return

        logger.info(f"Opening {side} | qty={qty} @ ~{price:.4f} | SL={sl:.4f} | TP={tp:.4f}")

        order_side = "BUY" if signal == 1 else "SELL"
        order = self.client.place_market_order(self.symbol, order_side, qty)
        if not order:
            logger.error("Entry order failed")
            return

        entry = float(order.get("avgPrice") or price) or price

        # SL/TP: exchange orders on live, software monitoring on testnet
        exit_side = "SELL" if signal == 1 else "BUY"
        if self._use_exchange_orders:
            self.client.place_stop_loss(self.symbol,   exit_side, qty, sl)
            self.client.place_take_profit(self.symbol, exit_side, qty, tp)
        else:
            # Software-side SL/TP (works on testnet)
            self._sltp[self.symbol] = {"sl": sl, "tp": tp, "side": side, "qty": qty}
            logger.info(f"Software SL/TP registered | SL={sl:.4f} TP={tp:.4f}")

        self._position_open_time = datetime.now()

        self.trade_history.append({
            "id":            order.get("orderId", ""),
            "time":          datetime.now().isoformat(),
            "symbol":        self.symbol,
            "side":          side,
            "quantity":      qty,
            "entry_price":   entry,
            "stop_loss":     sl,
            "take_profit":   tp,
            "ml_confidence": self.ml_confidence,
            "status":        "OPEN",
            "pnl":           0.0,
        })
        # Keep only last 100 trades in memory
        self.trade_history = self.trade_history[-100:]

    # ── Dashboard data ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        positions = self.client.get_positions(self.symbol)
        daily     = self.risk.get_daily_stats()
        return {
            "status":           self.status,
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "leverage":         self.leverage,
            "current_price":    self.client.get_price(self.symbol),
            "signal":           self.current_signal,
            "ml_confidence":    self.ml_confidence,
            "indicator_signal": self.ind_signal,
            "positions":        positions,
            "daily_pnl":        daily["daily_pnl"],
            "daily_loss":       daily["daily_loss"],
            "ml_trained":       self.strategy.is_trained,
            "last_update":      self.last_update.isoformat() if self.last_update else None,
            "error":            self.error_message,
        }


def _timeframe_to_seconds(tf: str) -> float:
    """Convert Binance timeframe string to seconds."""
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        return int(tf[:-1]) * units[tf[-1]]
    except (KeyError, ValueError):
        return 60
