"""Binance Futures API wrapper"""
import math
import os

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger

# Testnet endpoints
_TESTNET_BASE_URL   = "https://testnet.binancefuture.com"
_TESTNET_STREAM_URL = "wss://stream.binancefuture.com"


class BinanceClient:
    def __init__(self, testnet: bool = False):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")

        self.testnet = testnet
        if testnet:
            self.client = Client(
                api_key, api_secret,
                testnet=True,
            )
            # Override futures base URL to Binance Futures Testnet
            self.client.FUTURES_URL = _TESTNET_BASE_URL + "/fapi"
            logger.warning("⚠️  TESTNET MODE — ไม่ใช้เงินจริง")
        else:
            self.client = Client(api_key, api_secret)
            logger.info("🔴 LIVE MODE — ใช้เงินจริง")
        self._test_connection()

    def _test_connection(self):
        try:
            self.client.futures_ping()
            logger.info("Binance Futures connection OK")
        except BinanceAPIException as e:
            logger.error(f"Binance connection failed: {e}")
            raise

    # ── Balance ──────────────────────────────────────────────────────────────

    def get_futures_balance(self) -> float:
        """USDT available balance in futures wallet"""
        try:
            for b in self.client.futures_account_balance():
                if b["asset"] == "USDT":
                    return float(b["availableBalance"])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    def get_total_balance(self) -> float:
        """Total USDT wallet balance (includes unrealized PnL)"""
        try:
            account = self.client.futures_account()
            return float(account["totalWalletBalance"])
        except BinanceAPIException as e:
            logger.error(f"Error getting total balance: {e}")
            return 0.0

    # ── Market data ──────────────────────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """OHLCV klines as DataFrame indexed by open_time"""
        limit = min(limit, 1500)  # Binance Futures hard cap
        try:
            raw = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(
                raw,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            return df
        except BinanceAPIException as e:
            logger.error(f"Error getting klines: {e}")
            return pd.DataFrame()

    def get_price(self, symbol: str) -> float:
        """Current mark price"""
        try:
            return float(self.client.futures_mark_price(symbol=symbol)["markPrice"])
        except BinanceAPIException as e:
            logger.error(f"Error getting price: {e}")
            return 0.0

    # ── Positions & Orders ───────────────────────────────────────────────────

    def get_positions(self, symbol: str = None) -> list:
        """All open futures positions (positionAmt != 0)"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            return [p for p in positions if float(p["positionAmt"]) != 0]
        except BinanceAPIException as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_open_orders(self, symbol: str) -> list:
        try:
            return self.client.futures_get_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    # ── Configuration ────────────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            return True
        except BinanceAPIException as e:
            if e.code == -4046:  # Already set
                return True
            logger.error(f"Error setting margin type: {e}")
            return False

    # ── Order placement ──────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Market order. side = 'BUY' | 'SELL'"""
        try:
            order = self.client.futures_create_order(
                symbol=symbol, side=side, type="MARKET", quantity=quantity
            )
            logger.info(f"Market order: {side} {quantity} {symbol} → orderId={order.get('orderId')}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing market order: {e}")
            return {}

    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """STOP_MARKET order (stop loss). side = opposite of position."""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=round(stop_price, 2),
                reduceOnly=True,
            )
            logger.info(f"Stop loss placed: {side} {quantity} @ {stop_price:.2f}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing stop loss: {e}")
            return {}

    def place_take_profit(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """TAKE_PROFIT_MARKET order. side = opposite of position."""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=round(stop_price, 2),
                reduceOnly=True,
            )
            logger.info(f"Take profit placed: {side} {quantity} @ {stop_price:.2f}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing take profit: {e}")
            return {}

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"All orders cancelled for {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Error cancelling orders: {e}")
            return False

    def close_position(self, symbol: str, close_side: str, quantity: float) -> dict:
        """Close position with market order. close_side = opposite of position."""
        return self.place_market_order(symbol, close_side, quantity)

    # ── Symbol info ──────────────────────────────────────────────────────────

    def get_step_size(self, symbol: str) -> float:
        """Minimum quantity step size for the symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            return float(f["stepSize"])
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
        return 0.001

    def round_quantity(self, quantity: float, step_size: float) -> float:
        """Round quantity down to the nearest valid step size"""
        precision = int(round(-math.log10(step_size)))
        floored = quantity - (quantity % step_size)
        return round(floored, precision)
