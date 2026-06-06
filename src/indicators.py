"""Technical Indicators using the `ta` library"""
import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add all technical indicator columns to the OHLCV DataFrame.
    Drops NaN rows resulting from lookback windows.
    """
    cfg = config.get("strategy", {})
    df = df.copy()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Trend ────────────────────────────────────────────────────────────────
    df["ema_fast"]  = ta.trend.EMAIndicator(close, window=cfg.get("ema_fast",  3)).ema_indicator()
    df["ema_slow"]  = ta.trend.EMAIndicator(close, window=cfg.get("ema_slow",  8)).ema_indicator()
    df["ema_trend"] = ta.trend.EMAIndicator(close, window=cfg.get("ema_trend", 21)).ema_indicator()

    # Approximate VWAP using cumulative (tp × vol) / cumvol over rolling 20
    typical_price = (high + low + close) / 3
    df["vwap"] = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum()

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(close, window=cfg.get("rsi_period", 7)).rsi()

    macd = ta.trend.MACD(
        close,
        window_fast=cfg.get("macd_fast",   6),
        window_slow=cfg.get("macd_slow",  13),
        window_sign=cfg.get("macd_signal",  4),
    )
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=5, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── Volatility ───────────────────────────────────────────────────────────
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=7).average_true_range()

    bb = ta.volatility.BollingerBands(close, window=14, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"]   = bb.bollinger_pband()

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_ma = vol.rolling(20).mean()
    df["vol_ratio"]  = vol / vol_ma   # > spike_mult → volume surge
    df["vol_delta"]  = vol - vol.shift(1)  # volume momentum

    # ── Derived features ─────────────────────────────────────────────────────
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    df["ema_fast_slow_ratio"]  = df["ema_fast"]  / df["ema_slow"]  - 1
    df["price_ema_fast_ratio"] = close           / df["ema_fast"]  - 1
    df["price_ema_slow_ratio"] = close           / df["ema_slow"]  - 1
    df["price_vwap_ratio"]     = close           / df["vwap"]      - 1  # above/below VWAP

    # Candle body ratio (large body = strong directional move)
    df["body_ratio"] = (close - df["open"]).abs() / (high - low + 1e-9)

    df.dropna(inplace=True)
    return df


def get_signal_from_indicators(df: pd.DataFrame, config: dict) -> int:
    """
    Scalping rule-based signal.
    Returns: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
    """
    if df.empty or len(df) < 2:
        return 0

    cfg    = config.get("strategy", {})
    rsi_os = cfg.get("rsi_oversold",       30)
    rsi_ob = cfg.get("rsi_overbought",     70)
    vspike = cfg.get("volume_spike_mult", 1.5)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Filters ───────────────────────────────────────────────────────────────
    trend_up   = last["close"] > last["ema_trend"]
    trend_down = last["close"] < last["ema_trend"]

    # EMA cross or alignment
    ema_cross_up   = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
    ema_bullish    = last["ema_fast"] > last["ema_slow"]
    ema_bearish    = last["ema_fast"] < last["ema_slow"]

    # MACD momentum flip
    macd_up   = last["macd_diff"] > 0 and last["macd_diff"] > prev["macd_diff"]
    macd_down = last["macd_diff"] < 0 and last["macd_diff"] < prev["macd_diff"]

    # Price vs VWAP
    above_vwap = last["close"] > last["vwap"]
    below_vwap = last["close"] < last["vwap"]

    # Volume confirmation
    vol_confirmed = last["vol_ratio"] >= vspike

    rsi_ok_long  = last["rsi"] < rsi_ob
    rsi_ok_short = last["rsi"] > rsi_os

    # ── Scalp LONG: trend up + EMA bullish + MACD + above VWAP + volume ──────
    if (trend_up and ema_bullish and macd_up
            and above_vwap and vol_confirmed and rsi_ok_long):
        return 1

    # ── Scalp SHORT ───────────────────────────────────────────────────────────
    if (trend_down and ema_bearish and macd_down
            and below_vwap and vol_confirmed and rsi_ok_short):
        return -1

    return 0

