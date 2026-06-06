# crypto-scalping-bot

AI Futures Scalping Bot สำหรับ Binance USDT-M Futures
ใช้ Random Forest + Technical Indicators (EMA, RSI, MACD, VWAP) เทรดหลายเหรียญพร้อมกัน

---

## Features

- **Multi-symbol** — เทรดหลายเหรียญพร้อมกัน (BTCUSDT, ETHUSDT, SOLUSDT, ...)
- **AI Signal** — Random Forest + Indicators ต้องตรงกันก่อนเข้าเทรด
- **Auto Retrain** — model train ใหม่ทุก 6 ชั่วโมงด้วยข้อมูลสด
- **Software SL/TP** — ทำงานบน Testnet ได้ (ไม่ต้องใช้ exchange orders)
- **Backtesting** — ดาวน์โหลดข้อมูลย้อนหลัง 4 ปีจาก Binance Vision แล้วทดสอบ
- **Web Dashboard** — monitor signal, positions, P&L แบบ realtime
- **Risk Management** — position sizing, daily loss limit, time-stop

---

## Requirements

- Python 3.11+
- Binance Futures account (หรือ Testnet)

---

## Installation

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/crypto-scalping-bot.git
cd crypto-scalping-bot

# 2. สร้าง virtual environment (แนะนำ)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. ติดตั้ง dependencies
pip install -r requirements.txt
```

---

## Configuration

### 1. สร้างไฟล์ `.env`

```bash
copy .env.example .env
```

แก้ไขใส่ API Key:

```env
# Testnet → https://testnet.binancefuture.com (คลิก "API Key" มุมบนขวา)
# Live    → https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=true
```

### 2. ปรับ `config.yaml`

```yaml
trading:
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]   # เหรียญที่ต้องการ
  timeframe: 1m                           # 1m=scalping | 15m=swing
  leverage: 10
  testnet: true                           # false = เงินจริง

risk:
  max_position_pct: 0.02    # risk 2% ต่อ trade
  stop_loss_pct: 0.004      # SL 0.4%
  take_profit_ratio: 1.5    # TP 1.5:1
  max_daily_loss_pct: 0.06  # หยุดถ้าขาดทุน > 6%/วัน
```

---

## Usage

### Train ML Model ครั้งแรก

```bash
python main.py --mode train
```

ดึงข้อมูล 1,500 candle ล่าสุดจาก Binance แล้ว train Random Forest สำหรับทุกเหรียญ
โมเดลถูกบันทึกไว้ที่ `models/rf_<symbol>.pkl`

### รัน Bot + Dashboard

```bash
python main.py
```

เปิด browser ที่ **http://localhost:8080**

### รัน Bot เท่านั้น (ไม่มี dashboard)

```bash
python main.py --mode bot
```

### รัน Dashboard เท่านั้น (ไม่เทรด)

```bash
python main.py --mode dashboard
```

---

## Backtesting

ทดสอบย้อนหลังด้วยข้อมูล Historical จาก [data.binance.vision](https://data.binance.vision)

```bash
# ติดตั้ง dependencies เพิ่มเติม
pip install requests pyarrow matplotlib

# Backtest ทุกเหรียญใน config (ย้อนหลัง 4 ปี)
python backtest.py

# Backtest เหรียญเดียว
python backtest.py --symbol BTCUSDT

# พร้อม equity curve chart
python backtest.py --symbol SOLUSDT --plot

# กำหนด timeframe และจำนวนปี
python backtest.py --symbol BTCUSDT --interval 5m --years 2

# กำหนด starting balance
python backtest.py --balance 10000
```

ข้อมูลจะถูก cache ไว้ที่ `data/klines/*.parquet` — ครั้งถัดไปไม่ต้อง download ใหม่

**ตัวอย่าง output:**
```
────────────────────────────────────────────────────
  BACKTEST REPORT  BTCUSDT 1m
────────────────────────────────────────────────────
  Period          2022-06-01 → 2026-05-31
  Initial balance $10,000.00
  Final balance   $12,453.20
  Total return    +24.53%
────────────────────────────────────────────────────
  Trades          1,842
  Win rate        54.3%  (1000W / 842L)
  Avg win         +0.45%
  Avg loss        -0.28%
  Profit factor   1.72
  Max drawdown    18.4%
  Sharpe ratio    1.34
────────────────────────────────────────────────────
```

---

## Project Structure

```
crypto-scalping-bot/
├── main.py                  ← entry point (bot + dashboard)
├── backtest.py              ← backtest CLI
├── config.yaml              ← ตั้งค่าทั้งหมด
├── requirements.txt
├── .env.example
├── src/
│   ├── binance_client.py    ← Binance Futures API wrapper
│   ├── indicators.py        ← EMA, RSI, MACD, ATR, VWAP, Bollinger
│   ├── ml_strategy.py       ← Random Forest Classifier
│   ├── risk_manager.py      ← Position sizing, SL/TP, daily loss guard
│   ├── trader.py            ← Bot loop + order execution
│   ├── bot_manager.py       ← Multi-symbol manager
│   ├── dashboard.py         ← Flask REST API
│   ├── backtest.py          ← Backtesting engine
│   └── data_downloader.py   ← Binance Vision historical data
├── templates/
│   └── index.html           ← Web dashboard
├── models/                  ← ML models (*.pkl) saved here
├── data/                    ← Historical klines cache (*.parquet)
└── logs/                    ← Log files
```

---

## Deploy บน VPS (รันตลอด 24/7)

```bash
# ติดตั้ง screen
sudo apt install screen

# รัน bot ใน background session
screen -S tradebot
python main.py

# กด Ctrl+A แล้ว D เพื่อ detach (bot ยังรันอยู่)
# กลับมาดูด้วย: screen -r tradebot
```

หรือใช้ **Oracle Cloud Free Tier** / **Google Cloud e2-micro** (ฟรีตลอดชีพ)

---

## ⚠️ คำเตือน

- **ทดสอบบน Testnet ก่อนเสมอ** — ตั้ง `testnet: true` ใน config
- Backtest ผ่านไม่ได้การันตีผลจริง (past performance ≠ future results)
- Scalping 1m มี noise สูง — แนะนำ 200+ trades บน testnet ก่อน live
- ไม่ควรใช้ leverage สูงถ้าไม่เข้าใจความเสี่ยง
