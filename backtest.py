#!/usr/bin/env python3
"""
Backtest CLI
============
Usage examples:
  python backtest.py                          # all symbols in config, 4 years, 1m
  python backtest.py --symbol BTCUSDT         # single symbol
  python backtest.py --symbol SOLUSDT --years 2 --interval 5m
  python backtest.py --balance 10000          # starting balance
  python backtest.py --plot                   # show equity curve chart
  python backtest.py --no-cache               # re-download data (ignore cache)
"""
import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _setup_logging():
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _plot_equity(result, symbol: str):
    try:
        import matplotlib.pyplot as plt
        eq = result.equity_curve
        xs = list(range(len(eq)))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(xs, eq, linewidth=1, color="#3fb950")
        ax.fill_between(xs, eq, min(eq), alpha=0.1, color="#3fb950")
        ax.set_title(f"Equity Curve — {symbol}  |  Return: {result.total_return_pct:+.2f}%  |  Sharpe: {result.sharpe_ratio:.2f}")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Balance (USDT)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = Path(f"data/equity_{symbol}.png")
        plt.savefig(out, dpi=150)
        logger.info(f"Equity chart saved → {out}")
        plt.show()
    except ImportError:
        logger.warning("matplotlib not installed — run: pip install matplotlib")


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(description="AI Futures Backtester")
    parser.add_argument("--symbol",   default=None,         help="Symbol to backtest (default: all in config)")
    parser.add_argument("--interval", default=None,         help="Candle interval (default: from config)")
    parser.add_argument("--years",    type=int, default=4,  help="Years of history (default: 4)")
    parser.add_argument("--balance",  type=float, default=10_000.0, help="Starting balance in USDT")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--plot",     action="store_true",  help="Plot equity curve")
    parser.add_argument("--no-cache", action="store_true",  help="Re-download data")
    args = parser.parse_args()

    config   = _load_config(args.config)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    trading_cfg = config.get("trading", {})
    interval    = args.interval or trading_cfg.get("timeframe", "1m")
    symbols_all = trading_cfg.get("symbols") or [trading_cfg.get("symbol", "BTCUSDT")]
    symbols     = [args.symbol.upper()] if args.symbol else symbols_all

    # Override interval in config so indicators use correct parameters
    config["trading"]["timeframe"] = interval

    from src.data_downloader import load_historical
    from src.backtest        import Backtester, print_report

    backtester = Backtester(config, initial_balance=args.balance)

    all_results = []
    for symbol in symbols:
        logger.info(f"{'='*52}")
        logger.info(f"Backtesting {symbol} | {interval} | {args.years}y | ${args.balance:,.0f}")
        logger.info(f"{'='*52}")
        try:
            df = load_historical(
                symbol, interval,
                years=args.years,
                cache=not args.no_cache,
            )
            result = backtester.run(df, symbol)
            print_report(result)
            all_results.append(result)

            if args.plot:
                _plot_equity(result, symbol)

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            continue

    # ── Portfolio summary ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("=" * 52)
        print("  PORTFOLIO SUMMARY")
        print("=" * 52)
        for r in all_results:
            status = "✓" if r.total_return_pct > 0 else "✗"
            print(
                f"  {status} {r.symbol:<12} "
                f"return={r.total_return_pct:+6.1f}%  "
                f"WR={r.win_rate:.0f}%  "
                f"PF={r.profit_factor:.2f}  "
                f"DD={r.max_drawdown_pct:.1f}%  "
                f"Sharpe={r.sharpe_ratio:.2f}"
            )
        avg_return = sum(r.total_return_pct for r in all_results) / len(all_results)
        print(f"\n  Average return across {len(all_results)} symbols: {avg_return:+.2f}%")
        print("=" * 52)


if __name__ == "__main__":
    main()
