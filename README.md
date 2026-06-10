# crypto-scalping-bot

AI Futures Scalping Bot สำหรับ Binance USDT-M Futures — สถาปัตยกรรม **V2 production-grade** ตาม `docs/models_handbook_v2_production.md`

> สถานะปัจจุบัน: **Phase 1-3 เสร็จ** (Optuna HP search, Regime eval, Kelly sizing, Correlation guard, Ensemble RF+GBM+ET, Signal/ExecutionOrder interfaces, APV-PLN PyTorch model พร้อมใช้)
> Compliance vs Handbook V2: **~78%** สำหรับ scope scalper-only — เปิด live testnet ได้

---

## Features

### Core (Phase 0)
- **Multi-symbol** — เทรด 12 เหรียญพร้อมกัน (BTC/ETH/BNB/SOL/XRP/DOGE/ADA/AVAX/LINK/DOT/TRX/NEAR)
- **Software SL/TP** — ทำงานบน Testnet ได้ (ไม่ต้องใช้ exchange orders)
- **Backtesting** — ดาวน์โหลดข้อมูลย้อนหลัง 4 ปีจาก Binance Vision (เพิ่ม fee 0.04% + break-even SL)
- **Web Dashboard** — monitor signal, positions, P&L แบบ realtime
- **SQLite trade log** — WAL mode, persist trades + signals

### Phase 1 — Validation & Risk (handbook §4-§8)
- **Optuna HP search** — 50 trials/symbol (RF: n_estimators, max_depth, min_samples_leaf, max_features)
- **Multi-seed training** — seeds [42, 123, 456], เลือก best-by-val-Sharpe + WF variance check
- **Walk-forward validation** — 5 walks, Sharpe variance gate <30%
- **Regime-stratified eval** — แยก metric ตาม trending/ranging (ATR-based)
- **10 V2 validation gates** — Sharpe>1.2, Acc>55%, PF>1.5, MaxDD<20%, Calmar>1.0, Sortino>1.5, WinRate>45%, Payoff>1.0, OOS, WF-var
- **Kelly position sizing** — Half-Kelly capped at max_position_pct
- **Correlation guard** — block symbol ใหม่ถ้า corr ≥ 0.85 กับ position ที่เปิดอยู่
- **5-condition kill-switch** — Portfolio DD / Daily Loss / Rolling Sharpe / Feature drift KL / Exchange connectivity

### Phase 2 — Ensemble (handbook §9)
- **3-model ensemble** — RandomForest + GradientBoosting + ExtraTrees
- **Softmax weight init** จาก validation Sharpe + **EMA refresh** (90% old + 10% new)
- **Consensus voting** — `ensemble_min_agree=2` (majority)
- **Conflict resolution** — size scalar (conf>0.70→1.0, >0.60→0.7, else 0.3, halve เมื่อ |p-0.5|<0.10)
- **Label denoising** — drop samples ที่ |future_ret| < 0.5×ATR

### Phase 3 — Standardized Interfaces (handbook §14)
- **ModelInterface ABC** — `predict(features) → {prediction, confidence, latency_ms, version}`
- **Signal dataclass §14.2** — `to_json()` + `signal_id` + full attribution
- **ExecutionOrder dataclass §14.3** — `to_order_message()` (Binance-compatible)
- **APV-PLN model** — PriceCNN + VolumeCNN + Cross-Attention + 51-bin head + LUPI distillation (Oracle teacher) — torch optional

### Phase 4 — Monitoring & Retraining (handbook §12)
- **Feature drift detector** — KL divergence vs training stats + rolling window
- **Auto-retrain** — 5 triggers: scheduled (6h), Sharpe<0.5, drift>threshold, win_rate<40%, gate-fail
- **Rolling metrics** — Sharpe, Win rate, Payoff ratio (50-trade window)
- **Signal expiry** — discard signals older than 300s

---

## Requirements

- Python 3.11+ (ทดสอบกับ 3.14.5)
- Binance Futures account (หรือ Testnet)
- **Optional**: PyTorch + CUDA สำหรับ APV-PLN Phase 3 (เทสต์กับ RTX 3060 12GB)

---

## Installation

```bash
# 1. Clone repo
git clone https://github.com/khunjibna/scalpr-bn.git
cd scalpr-bn

# 2. สร้าง virtual environment (แนะนำ)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. (Optional) ติดตั้ง PyTorch สำหรับ APV-PLN
#    CUDA 12.6 (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu126
#    DirectML (AMD/Intel GPU on Windows)
pip install torch-directml
#    CPU only
pip install torch
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

### 2. ปรับ `config.yaml` (สำคัญ — preset สำหรับ portfolio $5,000)

```yaml
trading:
  symbols: [BTCUSDT, ETHUSDT, ...]   # 12 เหรียญ default
  timeframe: 1m                       # 1m=scalping
  leverage: 3                         # ลด 5→3 (DD 71%→~21%)
  testnet: true
  fee_rate: 0.0004                    # Binance taker fee

risk:
  max_position_pct: 0.01              # 1% × $5,000 = $50/trade
  stop_loss_pct: 0.005                # SL 0.5%
  take_profit_ratio: 1.0              # R:R 1:1
  breakeven_at_r: 0.5                 # เลื่อน SL ไป entry เมื่อ +0.5R
  max_daily_loss_pct: 0.03            # หยุดถ้าขาดทุน > 3%/วัน
  max_portfolio_dd: 0.15              # หยุดถ้า portfolio DD > 15%
  max_corr_threshold: 0.85            # บล็อก symbol ที่ correlate ≥ 0.85

ml:
  retrain_hours: 6
  confidence_threshold: 0.62          # min confidence threshold
  optuna_trials: 50                   # HP search trials per symbol
  ensemble_min_agree: 2               # consensus: 2/3 models must agree
  min_move_atr: 0.5                   # denoise: drop samples < 0.5×ATR move
  apv_pln:
    enabled: false                    # ตั้ง true เพื่อใช้ APV-PLN (ต้องมี torch)
    backend: cuda                     # cpu | cuda | directml
```

---

## Usage

### Train ML Models ครั้งแรก (12 symbols)

```bash
python main.py --mode train
```

ดึงข้อมูล 1,500 candle ล่าสุดจาก Binance → label denoise → Optuna HP (50 trials) → 3-seed training → ensemble fit → V2 validation gates → save `models/rf_<symbol>_ensemble.pkl`

ใช้เวลา ~5-10 นาทีสำหรับ 12 symbols.

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

ทดสอบย้อนหลังด้วยข้อมูล Historical จาก [data.binance.vision](https://data.binance.vision) — **มี fee + break-even SL realistic**

```bash
# Backtest ทุกเหรียญใน config (ย้อนหลัง 4 ปี)
python backtest.py

# Backtest เหรียญเดียว + equity curve chart
python backtest.py --symbol BTCUSDT --plot

# กำหนด timeframe + starting balance
python backtest.py --symbol BTCUSDT --interval 5m --years 2 --balance 10000
```

ข้อมูลจะถูก cache ไว้ที่ `data/klines/*.parquet` — ครั้งถัดไปไม่ต้อง download ใหม่

**ตัวอย่าง output (Phase 2 fixes applied):**
```
────────────────────────────────────────────────────
  BACKTEST REPORT  BTCUSDT 1m  (with fees + BE SL)
────────────────────────────────────────────────────
  Period          2022-06-01 → 2026-05-31
  Trades          1,842   Win rate  54.3%
  Profit factor   1.72    Max drawdown 18.4%
  Sharpe ratio    1.34    Total fees   $73.20
  Exit reasons    TP=42%  SL=35%  BE=15%  TIME=8%
────────────────────────────────────────────────────
```

---

## Project Structure

```
APP_TRADE/
├── main.py                    ← entry point (bot + dashboard + train)
├── backtest.py                ← backtest CLI
├── config.yaml                ← ตั้งค่าทั้งหมด (12 symbols default)
├── requirements.txt
├── .env.example
├── docs/
│   └── models_handbook_v2_production.md   ← 2030-line V2 spec (gitignored)
├── src/
│   ├── binance_client.py      ← Binance API wrapper + is_connected probe
│   ├── indicators.py          ← 20 features, schema v2.2
│   ├── ml_strategy.py         ← Training pipeline (Optuna+multi-seed+WF+gates)
│   ├── ensemble.py            ← 3-model ensemble (RF+GBM+ET) ⊂ ModelInterface
│   ├── interfaces.py          ← ModelInterface + Signal + ExecutionOrder (§14)
│   ├── apv_pln.py             ← PyTorch CNN+Attn+LUPI model (Phase 3, optional)
│   ├── risk_manager.py        ← Kelly sizing + 5-condition kill-switch
│   ├── trader.py              ← Bot loop (emits Signal dataclass per cycle)
│   ├── bot_manager.py         ← Multi-symbol + margin + correlation monitor
│   ├── database.py            ← SQLite WAL trade log
│   ├── dashboard.py           ← Flask REST API
│   ├── backtest.py            ← Backtest engine (fee + BE SL realistic)
│   └── data_downloader.py     ← Binance Vision historical data
├── templates/
│   └── index.html             ← Bootstrap 5 dashboard
├── models/                    ← rf_<symbol>_ensemble.pkl (gitignored)
├── data/
│   ├── klines/                ← Historical parquet cache (gitignored)
│   └── trades.db              ← SQLite trade log (gitignored)
└── logs/                      ← Daily logs (gitignored)
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

## V2 Compliance Status

| Section | Coverage |
|---|---|
| §3 Data versioning + drift | 78% (schema v2.2, KL drift; missing SHA256 dataset hash) |
| §4 Training (Optuna + multi-seed) | 70% (50 trials vs 200 spec; sklearn no early-stop) |
| §5 Walk-forward + regime eval | 90% (5 walks vs 7 spec; 2 regimes vs 6) |
| §6 10 validation gates | 80% (8/10 hard gates) |
| §7 Execution layer | 33% ⚠️ (only MARKET, no LIMIT/IOC/slippage tracking) |
| §8 Risk + kill-switch | 87% (5 conditions wired) |
| §9 Ensemble + consensus | 90% (3-model softmax+EMA) |
| §11 Live safeguards | 75% (min conf + signal expiry + multi-source agree) |
| §12 Monitoring + retrain | 86% (5 retrain triggers + drift detector) |
| §14 ModelInterface + Signal | 70% (interfaces wired into trader, ExecutionOrder pending) |
| §APV-PLN appendix | code complete, training pending |

ไม่อยู่ใน scope: §10 RL Market Maker, §13/§15 multi-archetype procedures

---

## ⚠️ คำเตือน

- **ทดสอบบน Testnet ก่อนเสมอ** — ตั้ง `testnet: true` ใน config
- Backtest ผ่านไม่ได้การันตีผลจริง (past performance ≠ future results)
- Scalping 1m มี noise สูง — แนะนำ 200+ trades บน testnet ก่อน live
- ไม่ควรใช้ leverage สูงถ้าไม่เข้าใจความเสี่ยง
- **Known issue**: software SL/TP เก็บใน RAM — ถ้า restart bot จะเกิด orphan position (กำลังจะ persist ลง DB ใน patch ถัดไป)

