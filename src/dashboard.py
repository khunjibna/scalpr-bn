"""Flask Web Dashboard"""
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from loguru import logger


def create_app(manager, config: dict) -> Flask:
    """
    manager: BotManager instance
    """
    app = Flask(__name__, template_folder="../templates")
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    CORS(app)

    # ── Pages ─────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Aggregate API ─────────────────────────────────────────────────────────

    @app.route("/api/status")
    def api_status():
        try:
            return jsonify(manager.get_all_status())
        except Exception as e:
            logger.error(f"Status API error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/trades")
    def api_trades():
        return jsonify(manager.get_all_trades())

    @app.route("/api/symbols")
    def api_symbols():
        return jsonify(list(manager.bots.keys()))

    @app.route("/api/chart/<symbol>")
    def api_chart(symbol: str):
        try:
            return jsonify(manager.get_chart(symbol.upper()))
        except Exception as e:
            logger.error(f"Chart API error: {e}")
            return jsonify([]), 500

    # ── Control: all bots ─────────────────────────────────────────────────────

    @app.route("/api/start", methods=["POST"])
    def api_start():
        manager.start_all()
        return jsonify({"message": "All bots started"})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        manager.stop_all()
        return jsonify({"message": "All bots stopping…"})

    @app.route("/api/train", methods=["POST"])
    def api_train():
        manager.train_all()
        return jsonify({"message": f"Training started for {len(manager.bots)} symbols"})

    # ── Control: per symbol ───────────────────────────────────────────────────

    @app.route("/api/start/<symbol>", methods=["POST"])
    def api_start_symbol(symbol: str):
        manager.start_symbol(symbol.upper())
        return jsonify({"message": f"{symbol.upper()} bot started"})

    @app.route("/api/stop/<symbol>", methods=["POST"])
    def api_stop_symbol(symbol: str):
        manager.stop_symbol(symbol.upper())
        return jsonify({"message": f"{symbol.upper()} bot stopping…"})

    @app.route("/api/train/<symbol>", methods=["POST"])
    def api_train_symbol(symbol: str):
        manager.train_symbol(symbol.upper())
        return jsonify({"message": f"Training started for {symbol.upper()}"})

    @app.route("/api/close_position/<symbol>", methods=["POST"])
    def api_close_position(symbol: str):
        sym = symbol.upper()
        try:
            bot = manager.bots.get(sym)
            if not bot:
                return jsonify({"error": f"Symbol {sym} not found"}), 404
            positions = bot.client.get_positions(sym)
            if not positions:
                return jsonify({"message": "No open positions"})
            for pos in positions:
                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue
                bot.client.cancel_all_orders(sym)
                close_side = "SELL" if amt > 0 else "BUY"
                bot.client.close_position(sym, close_side, abs(amt))
            return jsonify({"message": f"{sym} position(s) closed"})
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/close_all", methods=["POST"])
    def api_close_all():
        closed = []
        for sym, bot in manager.bots.items():
            positions = bot.client.get_positions(sym)
            for pos in positions:
                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue
                bot.client.cancel_all_orders(sym)
                close_side = "SELL" if amt > 0 else "BUY"
                bot.client.close_position(sym, close_side, abs(amt))
                closed.append(sym)
        return jsonify({"message": f"Closed positions: {closed or 'none'}"})

    return app


    # ── Pages ─────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── API ───────────────────────────────────────────────────────────────────

    @app.route("/api/status")
    def api_status():
        try:
            return jsonify(bot.get_status())
        except Exception as e:
            logger.error(f"Status API error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/trades")
    def api_trades():
        return jsonify(list(reversed(bot.trade_history[-50:])))

    @app.route("/api/chart")
    def api_chart():
        try:
            df = bot.client.get_klines(bot.symbol, bot.timeframe, limit=150)
            if df.empty:
                return jsonify([])
            df = df.reset_index()
            candles = [
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
            return jsonify(candles)
        except Exception as e:
            logger.error(f"Chart API error: {e}")
            return jsonify([]), 500

    # ── Control endpoints ─────────────────────────────────────────────────────

    @app.route("/api/start", methods=["POST"])
    def api_start():
        if bot.status == "running":
            return jsonify({"message": "Bot already running"})
        t = threading.Thread(target=bot.start, daemon=True)
        t.start()
        return jsonify({"message": "Bot started"})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        bot.stop()
        return jsonify({"message": "Bot stopping…"})

    @app.route("/api/train", methods=["POST"])
    def api_train():
        def _train():
            ml_cfg = config.get("ml", {})
            df = bot.client.get_klines(
                bot.symbol, bot.timeframe,
                limit=ml_cfg.get("lookback_candles", 3000),
            )
            if not df.empty:
                from .indicators import calculate_indicators
                df = calculate_indicators(df, config)
                bot.strategy.train_model(df)

        threading.Thread(target=_train, daemon=True).start()
        return jsonify({"message": "Training started in background"})

    @app.route("/api/close_position", methods=["POST"])
    def api_close_position():
        try:
            positions = bot.client.get_positions(bot.symbol)
            if not positions:
                return jsonify({"message": "No open positions"})
            for pos in positions:
                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue
                bot.client.cancel_all_orders(bot.symbol)
                close_side = "SELL" if amt > 0 else "BUY"
                bot.client.close_position(bot.symbol, close_side, abs(amt))
            return jsonify({"message": "Position(s) closed"})
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return jsonify({"error": str(e)}), 500

    return app
