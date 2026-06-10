"""SQLite persistence layer — trade history.

Schema
------
trades:
    uid           TEXT PK   – UUID generated at entry time
    order_id      TEXT      – Binance orderId (may be empty on testnet)
    time          TEXT      – ISO-8601 open time
    close_time    TEXT      – ISO-8601 close time (NULL while OPEN)
    symbol        TEXT
    side          TEXT      – LONG | SHORT
    quantity      REAL
    entry_price   REAL
    stop_loss     REAL
    take_profit   REAL
    ml_confidence REAL
    status        TEXT      – OPEN | STOP LOSS | TAKE PROFIT | TIME STOP | MANUAL
    pnl           REAL
"""
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_DEFAULT_PATH = "data/trades.db"

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    uid           TEXT PRIMARY KEY,
    order_id      TEXT DEFAULT '',
    time          TEXT NOT NULL,
    close_time    TEXT,
    symbol        TEXT NOT NULL,
    side          TEXT DEFAULT '',
    quantity      REAL DEFAULT 0.0,
    entry_price   REAL DEFAULT 0.0,
    stop_loss     REAL DEFAULT 0.0,
    take_profit   REAL DEFAULT 0.0,
    ml_confidence REAL DEFAULT 0.0,
    status        TEXT DEFAULT 'OPEN',
    pnl           REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_time   ON trades(time DESC);
"""


class TradeDB:
    """Thread-safe SQLite wrapper — single persistent connection + Lock."""

    def __init__(self, db_path: str = _DEFAULT_PATH):
        self._path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(_DDL)
        logger.info(f"TradeDB ready: {db_path}")

    def close(self):
        """Explicit shutdown — call on process exit if desired."""
        with self._lock:
            self._con.close()

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_trade(self, trade: dict) -> str:
        """Insert a new OPEN trade. Returns the uid (generates one if missing)."""
        uid = trade.get("uid") or str(uuid.uuid4())
        with self._lock:
            self._con.execute(
                """
                INSERT OR IGNORE INTO trades
                    (uid, order_id, time, symbol, side, quantity,
                     entry_price, stop_loss, take_profit, ml_confidence, status, pnl)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    uid,
                    str(trade.get("id", "")),
                    trade.get("time", datetime.now().isoformat()),
                    trade.get("symbol", ""),
                    trade.get("side", ""),
                    float(trade.get("quantity", 0.0)),
                    float(trade.get("entry_price", 0.0)),
                    float(trade.get("stop_loss", 0.0)),
                    float(trade.get("take_profit", 0.0)),
                    float(trade.get("ml_confidence", 0.0)),
                    trade.get("status", "OPEN"),
                    float(trade.get("pnl", 0.0)),
                ),
            )
            self._con.commit()
        return uid

    def close_trade(self, uid: str, status: str, pnl: float) -> None:
        """Update status + pnl + close_time when a position is closed."""
        with self._lock:
            self._con.execute(
                "UPDATE trades SET status=?, pnl=?, close_time=? WHERE uid=?",
                (status, float(pnl), datetime.now().isoformat(), uid),
            )
            self._con.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_trades(self, limit: int = 100, symbol: str | None = None) -> list[dict]:
        """Return recent trades as list[dict], newest first."""
        with self._lock:
            if symbol:
                rows = self._con.execute(
                    "SELECT * FROM trades WHERE symbol=? ORDER BY time DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = self._con.execute(
                    "SELECT * FROM trades ORDER BY time DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self, symbol: str | None = None) -> float:
        """Sum of realized PnL for closed trades opened today (UTC date)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if symbol:
                row = self._con.execute(
                    "SELECT COALESCE(SUM(pnl),0) FROM trades "
                    "WHERE status != 'OPEN' AND symbol=? AND time LIKE ?",
                    (symbol, f"{today}%"),
                ).fetchone()
            else:
                row = self._con.execute(
                    "SELECT COALESCE(SUM(pnl),0) FROM trades "
                    "WHERE status != 'OPEN' AND time LIKE ?",
                    (f"{today}%",),
                ).fetchone()
        return float(row[0]) if row else 0.0

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        """Return all currently OPEN trades (for position recovery on restart)."""
        with self._lock:
            if symbol:
                rows = self._con.execute(
                    "SELECT * FROM trades WHERE status='OPEN' AND symbol=? ORDER BY time DESC",
                    (symbol,),
                ).fetchall()
            else:
                rows = self._con.execute(
                    "SELECT * FROM trades WHERE status='OPEN' ORDER BY time DESC"
                ).fetchall()
        return [dict(r) for r in rows]
