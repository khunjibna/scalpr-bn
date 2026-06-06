"""
Download historical klines from https://data.binance.vision
Supports Futures UM (USDT-M) monthly files.

URL pattern:
  https://data.binance.vision/data/futures/um/monthly/klines
    /{symbol}/{interval}/{symbol}-{interval}-{YYYY}-{MM}.zip
"""
import io
import os
import zipfile
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
DATA_DIR = Path("data/klines")


def _zip_url(symbol: str, interval: str, year: int, month: int) -> str:
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{fname}"


def _download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
    url  = _zip_url(symbol, interval, year, month)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, header=None)

    df.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df


def load_historical(
    symbol: str,
    interval: str,
    years: int = 4,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download and concatenate `years` worth of monthly klines.
    Caches each month as a parquet file under data/klines/.
    Returns a single DataFrame sorted by time.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today     = date.today()
    end_year  = today.year
    end_month = today.month - 1 or 12  # last completed month
    if today.month == 1:
        end_year -= 1

    # Build list of (year, month) going back `years`
    months: list[tuple[int, int]] = []
    y, m = end_year, end_month
    for _ in range(years * 12):
        months.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1

    frames: list[pd.DataFrame] = []
    total = len(months)

    for i, (yr, mo) in enumerate(reversed(months), 1):
        cache_path = DATA_DIR / f"{symbol}_{interval}_{yr}_{mo:02d}.parquet"

        if cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            frames.append(df)
            logger.debug(f"Cache hit: {cache_path.name}")
            continue

        logger.info(f"Downloading {symbol} {interval} {yr}-{mo:02d}  ({i}/{total})")
        try:
            df = _download_month(symbol, interval, yr, mo)
            if df.empty:
                logger.warning(f"No data for {symbol} {yr}-{mo:02d} (skipped)")
                continue
            if cache:
                df.to_parquet(cache_path)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed {symbol} {yr}-{mo:02d}: {e}")

    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval}")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    logger.info(f"Loaded {len(combined):,} candles for {symbol} {interval} ({years}y)")
    return combined
