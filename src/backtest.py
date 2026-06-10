"""
Backtesting Engine
==================
Walks through historical OHLCV bar-by-bar using the same
indicators + ML strategy as the live bot.

Train/test split:
  - First 70%  → train ML model
  - Last  30%  → out-of-sample test (walk-forward friendly)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .indicators import calculate_indicators, get_signal_from_indicators
from .ml_strategy import MLStrategy


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:    pd.Timestamp
    symbol:        str
    side:          str          # LONG | SHORT
    entry_price:   float
    qty:           float
    stop_loss:     float
    take_profit:   float
    ml_confidence: float
    initial_sl:    float = 0.0
    be_armed:      bool  = False
    exit_time:     Optional[pd.Timestamp] = None
    exit_price:    Optional[float]        = None
    exit_reason:   str                    = ""
    pnl:           float                  = 0.0
    pnl_pct:       float                  = 0.0


@dataclass
class BacktestResult:
    symbol:          str
    interval:        str
    start:           str
    end:             str
    initial_balance: float
    final_balance:   float
    total_return_pct: float
    trades:          list[Trade]   = field(default_factory=list)

    # Computed metrics (filled by analyse())
    n_trades:         int   = 0
    n_wins:           int   = 0
    n_losses:         int   = 0
    win_rate:         float = 0.0
    avg_win_pct:      float = 0.0
    avg_loss_pct:     float = 0.0
    profit_factor:    float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio:     float = 0.0
    total_fees:       float = 0.0
    exit_reasons:     dict  = field(default_factory=dict)
    equity_curve:     list  = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, config: dict, initial_balance: float = 10_000.0):
        self.config          = config
        self.initial_balance = initial_balance
        self.risk_cfg        = config.get("risk", {})
        self.trading_cfg     = config.get("trading", {})
        self.leverage        = self.trading_cfg.get("leverage", 10)
        # Binance USDT-M Futures taker fee ≈ 0.04%; round-trip = entry + exit.
        # Override via trading.fee_rate in config.yaml if needed.
        self.fee_rate        = float(self.trading_cfg.get("fee_rate", 0.0004))

    def run(self, df_raw: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Run backtest on a full historical DataFrame.
        df_raw must have OHLCV columns (no indicators yet).
        """
        logger.info(f"Computing indicators for {symbol} ({len(df_raw):,} candles)…")
        df = calculate_indicators(df_raw, self.config)
        if df.empty:
            raise ValueError("Indicator calculation returned empty DataFrame")

        # ── Train / test split ────────────────────────────────────────────────
        split = int(len(df) * 0.70)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        logger.info(f"Train: {len(train_df):,} candles | Test: {len(test_df):,} candles")
        logger.info(f"Train period: {train_df.index[0].date()} → {train_df.index[-1].date()}")
        logger.info(f"Test  period: {test_df.index[0].date()}  → {test_df.index[-1].date()}")

        ml = MLStrategy(self.config)
        ml.ml_cfg["model_path"] = f"models/bt_{symbol.lower()}.pkl"
        ok = ml.train_model(train_df)
        if not ok:
            logger.warning("ML training failed — running indicator-only signals")

        # ── Walk through test bars ────────────────────────────────────────────
        balance     = self.initial_balance
        equity      = [balance]
        open_trade: Optional[Trade] = None
        trades:     list[Trade]     = []
        entry_bar_i: int            = 0   # bar index when position opened

        sl_pct    = self.risk_cfg.get("stop_loss_pct",    0.004)
        rr        = self.risk_cfg.get("take_profit_ratio", 1.5)
        risk_frac = self.risk_cfg.get("max_position_pct", 0.02)
        be_r      = float(self.risk_cfg.get("breakeven_at_r", 0.0))
        max_hold  = self.config.get("strategy", {}).get("max_hold_candles", 10)
        conf_thr  = self.config.get("ml", {}).get("confidence_threshold", 0.60)

        # Pre-extract numpy arrays for fast bar access
        closes  = test_df["close"].values
        highs   = test_df["high"].values
        lows    = test_df["low"].values
        index   = test_df.index
        n_bars  = len(test_df)
        log_every = max(1, n_bars // 20)   # progress every 5%

        for i in range(1, n_bars):
            if i % log_every == 0:
                logger.info(f"Simulating {symbol}: {i/n_bars*100:.0f}% ({i:,}/{n_bars:,} bars) | trades={len(trades)} balance=${balance:,.0f}")

            price = closes[i]
            high  = highs[i]
            low   = lows[i]
            ts    = index[i]

            # ── Check open trade ──────────────────────────────────────────────
            if open_trade is not None:
                t   = open_trade                # Break-even SL move: lock in zero loss after +be_r · R move
                if be_r > 0 and not t.be_armed:
                    r_dist = abs(t.entry_price - t.initial_sl)
                    if r_dist > 0:
                        favourable = (high - t.entry_price) if t.side == "LONG" \
                                     else (t.entry_price - low)
                        if favourable >= be_r * r_dist:
                            t.stop_loss = t.entry_price
                            t.be_armed  = True   
                            hit = None
                if t.side == "LONG":
                    if low  <= t.stop_loss:    hit = ("SL", t.stop_loss)
                    elif high >= t.take_profit: hit = ("TP", t.take_profit)
                else:
                    if high >= t.stop_loss:    hit = ("SL", t.stop_loss)
                    elif low  <= t.take_profit: hit = ("TP", t.take_profit)

                # Time-stop — use bar index diff (O(1))
                if hit is None and max_hold > 0 and (i - entry_bar_i) >= max_hold:
                    hit = ("TIME", price)

                if hit:
                    reason, exit_px = hit
                    pnl = ((exit_px - t.entry_price) if t.side == "LONG"
                           else (t.entry_price - exit_px)) * t.qty * self.leverage
                    # Round-trip taker fees (entry notional + exit notional) × fee_rate
                    fee = (t.entry_price + exit_px) * t.qty * self.fee_rate
                    pnl -= fee
                    pnl_pct = pnl / balance * 100
                    t.exit_time   = ts
                    t.exit_price  = exit_px
                    t.exit_reason = reason
                    t.pnl         = pnl
                    t.pnl_pct     = pnl_pct
                    balance      += pnl
                    balance       = max(balance, 0.01)
                    open_trade    = None
                    trades.append(t)

                equity.append(balance)
                continue

            # ── Generate signal (every bar, no slicing overhead) ──────────────
            window = test_df.iloc[max(0, i - 100): i + 1]
            ind_signal = get_signal_from_indicators(window, self.config)
            pred = ml.predict(window)
            # Back-compat: ensemble returns 3-tuple, legacy returns 2-tuple
            if len(pred) == 3:
                ml_signal, confidence, _size_scalar = pred
            else:
                ml_signal, confidence = pred

            signal = ml_signal if (ind_signal == ml_signal != 0) else 0
            if signal == 0 or confidence < conf_thr:
                equity.append(balance)
                continue

            # ── Size the trade ────────────────────────────────────────────────
            side    = "LONG" if signal == 1 else "SHORT"
            sl_dist = price * sl_pct
            sl  = price - sl_dist if side == "LONG" else price + sl_dist
            tp  = price + sl_dist * rr if side == "LONG" else price - sl_dist * rr

            risk_amt = balance * risk_frac
            qty      = risk_amt / sl_dist
            margin   = qty * price / self.leverage
            if margin > balance * 0.20:
                qty = (balance * 0.20 * self.leverage) / price

            if qty * price < 5:
                equity.append(balance)
                continue

            open_trade = Trade(
                entry_time=ts, symbol=symbol, side=side,
                entry_price=price, qty=qty,
                stop_loss=sl, take_profit=tp,
                initial_sl=sl,
                ml_confidence=confidence,
            )
            entry_bar_i = i
            equity.append(balance)

        # Force-close any open position at last bar
        if open_trade is not None:
            last_price = test_df["close"].iloc[-1]
            t = open_trade
            pnl = ((last_price - t.entry_price) if t.side == "LONG"
                   else (t.entry_price - last_price)) * t.qty * self.leverage
            fee = (t.entry_price + last_price) * t.qty * self.fee_rate
            pnl -= fee
            t.exit_time   = test_df.index[-1]
            t.exit_price  = last_price
            t.exit_reason = "END"
            t.pnl         = pnl
            t.pnl_pct     = pnl / balance * 100
            balance      += pnl
            trades.append(t)

        result = BacktestResult(
            symbol          = symbol,
            interval        = self.trading_cfg.get("timeframe", "1m"),
            start           = str(test_df.index[0].date()),
            end             = str(test_df.index[-1].date()),
            initial_balance = self.initial_balance,
            final_balance   = balance,
            total_return_pct= (balance - self.initial_balance) / self.initial_balance * 100,
            trades          = trades,
            equity_curve    = equity,
        )
        return _analyse(result)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _analyse(r: BacktestResult) -> BacktestResult:
    trades = r.trades
    if not trades:
        return r

    pnls     = [t.pnl for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p < 0]

    r.n_trades    = len(trades)
    r.n_wins      = len(wins)
    r.n_losses    = len(losses)
    r.win_rate    = len(wins) / len(trades) * 100 if trades else 0
    r.avg_win_pct = np.mean([t.pnl_pct for t in trades if t.pnl > 0]) if wins   else 0
    r.avg_loss_pct= np.mean([t.pnl_pct for t in trades if t.pnl < 0]) if losses else 0

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    r.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Fee summary (sum of entry+exit fees; estimated from notional & fee_rate)
    # Each trade's fee is implicit in pnl now; reconstruct for transparency.
    fee_total = 0.0
    for t in trades:
        if t.exit_price is None:
            continue
        # Approx — mirrors the deduction made during simulation
        fee_total += (t.entry_price + t.exit_price) * t.qty * 0.0004
    r.total_fees = fee_total

    # Exit reason distribution (SL / TP / TIME / END)
    from collections import Counter
    r.exit_reasons = dict(Counter(t.exit_reason for t in trades))

    # Max drawdown
    eq  = np.array(r.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / peak * 100
    r.max_drawdown_pct = float(dd.max())

    # Sharpe (annualised, assumes 1m candles → 525,600 bars/year)
    returns = np.diff(eq) / eq[:-1]
    if returns.std() > 0:
        bars_per_year = 525_600
        r.sharpe_ratio = float(returns.mean() / returns.std() * math.sqrt(bars_per_year))

    return r


def print_report(r: BacktestResult):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT  {r.symbol} {r.interval}")
    print(sep)
    print(f"  Period          {r.start} → {r.end}")
    print(f"  Initial balance ${r.initial_balance:,.2f}")
    print(f"  Final balance   ${r.final_balance:,.2f}")
    print(f"  Total return    {r.total_return_pct:+.2f}%")
    print(sep)
    print(f"  Trades          {r.n_trades}")
    print(f"  Win rate        {r.win_rate:.1f}%  ({r.n_wins}W / {r.n_losses}L)")
    print(f"  Avg win         {r.avg_win_pct:+.2f}%")
    print(f"  Avg loss        {r.avg_loss_pct:+.2f}%")
    print(f"  Profit factor   {r.profit_factor:.2f}")
    print(f"  Max drawdown    {r.max_drawdown_pct:.2f}%")
    print(f"  Sharpe ratio    {r.sharpe_ratio:.2f}")
    print(sep)

    # Exit reason breakdown
    from collections import Counter
    reasons = Counter(t.exit_reason for t in r.trades)
    print("  Exit reasons:")
    for reason, cnt in sorted(reasons.items()):
        print(f"    {reason:<10} {cnt}")
    print(sep + "\n")
