"""Manages multiple single-symbol TradingBot instances"""
import copy
import os
import threading

import numpy as np
from loguru import logger

from .binance_client import BinanceClient
from .database import TradeDB
from .indicators import calculate_indicators
from .trader import TradingBot


class BotManager:
    def __init__(self, config: dict):
        self.config = config
        trading_cfg = config.get("trading", {})

        # Support both `symbols` (list) and legacy `symbol` (string)
        raw = trading_cfg.get("symbols") or [trading_cfg.get("symbol", "BTCUSDT")]
        symbols: list[str] = [s.upper() for s in raw]

        use_testnet = (
            trading_cfg.get("testnet", False)
            or os.getenv("USE_TESTNET", "false").lower() == "true"
        )

        # One shared Binance client for all bots
        sim_balance = trading_cfg.get("simulated_balance") or None
        self._client = BinanceClient(testnet=use_testnet, simulated_balance=sim_balance)

        # Shared SQLite DB (all bots write to the same file)
        db_path = config.get("database", {}).get("path", "data/trades.db")
        self._db = TradeDB(db_path)

        self.bots: dict[str, TradingBot] = {}
        for symbol in symbols:
            sym_config = copy.deepcopy(config)
            sym_config["trading"]["symbol"] = symbol
            # Each bot gets its own ML model path so models don't clash
            sym_config["ml"]["model_path"] = f"models/rf_{symbol.lower()}.pkl"
            self.bots[symbol] = TradingBot(sym_config, client=self._client, db=self._db)
            logger.info(f"Bot registered: {symbol}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_all(self):
        for symbol, bot in self.bots.items():
            t = threading.Thread(target=bot.start, daemon=True, name=f"Bot-{symbol}")
            t.start()
        # Background thread: keep free_margin updated across all bots
        threading.Thread(target=self._margin_updater, daemon=True,
                         name="MarginUpdater").start()

    def stop_all(self):
        for bot in self.bots.values():
            bot.stop()

    def _margin_updater(self):
        """Background thread: compute portfolio free margin and push to all bots."""
        import time
        _last_corr_check = 0.0
        _CORR_INTERVAL   = 60.0   # re-check correlations every 60 s
        max_corr = self.config.get("risk", {}).get("max_corr_threshold", 0.85)

        while True:
            try:
                balance = self._client.get_futures_balance()
                # Sum margin in use across all open positions
                used = 0.0
                open_symbols: list[str] = []
                for bot in self.bots.values():
                    for pos in self._client.get_positions(bot.symbol):
                        if abs(float(pos.get("positionAmt", 0))) > 0:
                            notional = abs(float(pos.get("notional", 0)))
                            leverage = float(pos.get("leverage", 1)) or 1
                            used += notional / leverage
                            open_symbols.append(bot.symbol)
                free = max(0.0, balance - used)
                for bot in self.bots.values():
                    bot.free_margin = free
                logger.debug(
                    f"Portfolio margin | balance={balance:.2f} "
                    f"used={used:.2f} free={free:.2f}"
                )

                # ── Correlation position limit (§8.1) ────────────────────────────
                now = time.time()
                if open_symbols and (now - _last_corr_check) >= _CORR_INTERVAL:
                    _last_corr_check = now
                    returns: dict[str, np.ndarray] = {}
                    for sym in set(open_symbols) | set(self.bots.keys()):
                        try:
                            df = self._client.get_klines(sym, "1m", limit=60)
                            if not df.empty:
                                returns[sym] = df["close"].pct_change().dropna().values
                        except Exception:
                            pass
                    corr_blocked: set[str] = set()
                    for sym in self.bots:
                        if sym in open_symbols or sym not in returns:
                            continue
                        for open_sym in open_symbols:
                            if open_sym not in returns:
                                continue
                            r1, r2 = returns[sym], returns[open_sym]
                            min_len = min(len(r1), len(r2))
                            if min_len < 20:
                                continue
                            corr = float(np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1])
                            if abs(corr) >= max_corr:
                                corr_blocked.add(sym)
                                logger.debug(
                                    f"Corr block: {sym} ↔ {open_sym} "
                                    f"corr={corr:.2f} ≥ {max_corr}"
                                )
                                break
                    for bot in self.bots.values():
                        bot.corr_blocked = bot.symbol in corr_blocked

            except Exception as e:
                logger.warning(f"MarginUpdater error: {e}")
            time.sleep(5)  # update every 5 s

    def start_symbol(self, symbol: str):
        bot = self.bots.get(symbol)
        if bot and bot.status != "running":
            threading.Thread(target=bot.start, daemon=True, name=f"Bot-{symbol}").start()

    def stop_symbol(self, symbol: str):
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()

    # ── Data ──────────────────────────────────────────────────────────────────

    def get_all_status(self) -> dict:
        """Aggregated status for dashboard (single balance API call)."""
        balance       = self._client.get_futures_balance()
        total_balance = self._client.get_total_balance()

        per_symbol: dict[str, dict] = {}
        running_count    = 0
        any_halted       = False
        halt_reasons     = []

        # Daily P&L from DB (survives restarts, includes all closed trades today)
        sym_pnls         = {sym: self._db.get_daily_pnl(sym) for sym in self.bots}
        total_daily_pnl  = sum(sym_pnls.values())
        total_daily_loss = sum(abs(v) for v in sym_pnls.values() if v < 0)

        for symbol, bot in self.bots.items():
            s = bot.get_status()
            per_symbol[symbol]    = s
            if s.get("status") == "running":
                running_count += 1
            if s.get("trading_halted"):
                any_halted = True
                reason = s.get("halt_reason", "")
                if reason:
                    halt_reasons.append(f"{symbol}: {reason}")

        return {
            "balance":       balance,
            "total_balance": total_balance,
            "daily_pnl":     total_daily_pnl,
            "daily_loss":    total_daily_loss,
            "symbols":       per_symbol,
            "bot_count":     len(self.bots),
            "running_count": running_count,
            "any_halted":    any_halted,
            "halt_reason":   " | ".join(halt_reasons),
        }

    def get_all_trades(self) -> list:
        """Fetch recent trades from SQLite (persists across restarts)."""
        return self._db.get_trades(limit=100)

    def get_chart(self, symbol: str) -> list:
        bot = self.bots.get(symbol)
        if not bot:
            return []
        df = self._client.get_klines(symbol, bot.timeframe, limit=150)
        if df.empty:
            return []
        df = df.reset_index()
        return [
            {
                "time":   int(row["open_time"].timestamp()),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in df.iterrows()
        ]

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self):
        """Retrain ML model for every symbol in background threads."""
        def _train(bot: TradingBot):
            ml_cfg = self.config.get("ml", {})
            df = self._client.get_klines(
                bot.symbol, bot.timeframe,
                limit=ml_cfg.get("lookback_candles", 1500),
            )
            if not df.empty:
                df = calculate_indicators(df, self.config)
                bot.strategy.train_model(df)

        for bot in self.bots.values():
            threading.Thread(target=_train, args=(bot,), daemon=True).start()

    def train_symbol(self, symbol: str):
        bot = self.bots.get(symbol)
        if not bot:
            return
        def _train():
            ml_cfg = self.config.get("ml", {})
            df = self._client.get_klines(
                bot.symbol, bot.timeframe,
                limit=ml_cfg.get("lookback_candles", 1500),
            )
            if not df.empty:
                df = calculate_indicators(df, self.config)
                bot.strategy.train_model(df)
        threading.Thread(target=_train, daemon=True).start()
