#!/usr/bin/env python3
"""
AI Futures Trading Bot — Binance
=================================
WARNING: This bot trades with REAL money.
         Always test with a small balance first.
         Past performance does not guarantee future results.

Usage:
  python main.py              # Start bot + dashboard together
  python main.py --mode bot   # Bot only (no dashboard)
  python main.py --mode dashboard  # Dashboard only (bot not running)
  python main.py --mode train      # Train ML model once and exit
"""
import argparse
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="00:00", retention="14 days",
               level="DEBUG", encoding="utf-8")


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _check_env():
    import os
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        logger.error("Set BINANCE_API_KEY and BINANCE_API_SECRET in a .env file")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AI Futures Trading Bot")
    parser.add_argument("--mode",   choices=["all", "bot", "dashboard", "train"],
                        default="all", help="Run mode (default: all)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    args = parser.parse_args()

    _setup_logging()
    _check_env()

    config = _load_config(args.config)
    import os
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data",   exist_ok=True)

    from src.bot_manager import BotManager
    from src.dashboard  import create_app

    manager = BotManager(config)

    # ── Train mode ────────────────────────────────────────────────────────────
    if args.mode == "train":
        logger.info("Downloading data and training models for all symbols…")
        from src.indicators import calculate_indicators
        ml_cfg = config.get("ml", {})
        trading_cfg = config.get("trading", {})
        symbols = trading_cfg.get("symbols") or [trading_cfg.get("symbol", "BTCUSDT")]
        timeframe = trading_cfg.get("timeframe", "15m")
        limit = ml_cfg.get("lookback_candles", 1500)
        any_ok = False
        for sym in symbols:
            df = manager._client.get_klines(sym, timeframe, limit=limit)
            if df.empty:
                logger.error(f"Could not fetch klines for {sym}")
                continue
            df = calculate_indicators(df, config)
            bot = manager.bots.get(sym)
            if bot:
                ok = bot.strategy.train_model(df)
                any_ok = any_ok or ok
        sys.exit(0 if any_ok else 1)

    # ── Bot thread ────────────────────────────────────────────────────────────
    if args.mode in ("all", "bot"):
        manager.start_all()
        logger.info(f"Started {len(manager.bots)} trading bots")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if args.mode in ("all", "dashboard"):
        app  = create_app(manager, config)
        dcfg = config.get("dashboard", {})
        host = dcfg.get("host", "0.0.0.0")
        port = dcfg.get("port", 8080)
        logger.info(f"Dashboard → http://localhost:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    else:
        # bot-only: keep main thread alive
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            manager.stop_all()
            logger.info("Shutting down…")


if __name__ == "__main__":
    main()
