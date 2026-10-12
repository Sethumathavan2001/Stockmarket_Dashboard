# Stock Market Dashboard

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.2-092E20?logo=django&logoColor=white)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-6001D2)
![Database](https://img.shields.io/badge/DB-SQLite-003B57?logo=sqlite&logoColor=white)

A Django web application for researching and practising **intraday trading on NSE stocks**.
It collects daily and 5-minute price data, screens and analyses ~2,000 companies, predicts
price moves with machine learning, and offers a full **paper-trading** simulator, in both
live and historical-replay modes. Six intraday strategies are shown on the paper-trading chart
with their historical results.

> **Disclaimer:** This project is for education and research. It does not execute real trades and
> nothing in it is investment advice. Back-test results describe the past only.

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Loading market data](#loading-market-data)
- [Research scripts](#research-scripts)
- [Configuration](#configuration)
- [Running the tests](#running-the-tests)
- [Known limitations](#known-limitations)

---

## Features

| Page | URL | What it does |
|---|---|---|
| **Home** | `/` | Companies by sector and industry. Click a bar of the industry chart to list its companies. |
| **Daily Data** | `/daily/` | Daily OHLC history of a company. |
| **5 Minutes** | `/5min/` | 5-minute candle history of a company. |
| **Live** | `/live/` | Live NSE gainers, losers, open-interest spurts and previous-day high/low breakouts. Click a symbol to open it in Paper Trading. |
| **Analysis** | `/analysis/` | Screener of yearly / monthly / weekly lows and highs, with filters and a background **Refresh** (progress bar). |
| **Chart** | `/chart/` | Interactive intraday candlestick charts. |
| **Daily Data Scraper** | `/daily-data/` | Downloads daily and 5-minute data from Yahoo Finance with a live progress bar; per-symbol errors are written to a log file. |
| **Stock Prediction** | `/stock_prediction/` | XGBoost models (walk-forward validated) that forecast the next 5-minute moves. |
| **Candle Predictor** | `/candle-predictor/` | Candlestick-pattern detection with a projection to 15:15. |
| **Paper Trading** | `/paper-trading/` | Virtual-money trading in **Live** or **Old** (replay a past day) mode. See below. |

### Paper trading

- Market orders, long and short, **Intraday** (auto square-off) or **Delivery** products, wallet with blocked margin.
- Stop loss and target on every position, monitored automatically across all held stocks.
- **Old mode** replays any stored day candle by candle at an adjustable speed; positions resume where you left off.
- **Strategy signals:** six intraday setups, each back-tested on the stock's own previous 60 days:
  Opening Range Breakout, VWAP Trend Pullback, RSI Reversal at S/R, Inside-Bar Breakout,
  Breakout and Retest, Supertrend + ADX. Entries and exits are marked on the chart, and a
  **Use** button fills the order ticket with the side, stop loss and target.
- Direct links: `/paper-trading/<CODE>/live/` and `/paper-trading/<CODE>/old/<YYYY-MM-DD>/`.

---

## Tech stack

- **Backend:** Python 3.13, Django 4.2, Django REST Framework
- **Data:** yfinance (Yahoo Finance), NSE India and Chartink public endpoints, pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn, joblib
- **Frontend:** Django templates, TradingView Lightweight Charts, Chart.js, ApexCharts, Bootstrap
- **Storage:** SQLite (development)
- **Reports:** openpyxl (Excel)

---

## Project structure

```
sethu_project/
├── manage.py
├── requirements.txt
├── sethu_project/            # Django settings and root URLs
├── market/                   # companies, daily / 5-minute data, home, analysis, live, chart, scraper
│   ├── models.py             # Company, DailyData, FiveMinuteData
│   ├── analysis_engine.py    # low / high analysis behind /analysis/
│   └── templates/market/     # base.html, shared navbar, pages
├── prediction/               # XGBoost price prediction (training, API, page)
├── patterns/                 # candlestick pattern detection and projection
├── paper_trading/            # paper-trading engine, API, page
│   ├── services.py           # orders, positions, wallet, live and replay prices
│   └── signals.py            # the six intraday strategies and their back-test
├── script/                   # standalone research scripts
│   └── output/               # their Excel reports (not committed)
├── static/                   # shared CSS
└── ml_models/                # trained models (not committed)
```

---

## Getting started

### Prerequisites

- Python 3.13 (3.11+ should work)
- Git

### Installation

```bash
git clone https://github.com/Sethumathavan2001/Stockmarket_Dashboard.git
cd Stockmarket_Dashboard

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### Database

Migrations are not committed, so create them once, then apply them:

```bash
python manage.py makemigrations market prediction patterns paper_trading
python manage.py migrate
python manage.py createsuperuser      # optional, for /admin/
```

### Run

```bash
python manage.py runserver
```

Open <http://127.0.0.1:8000/>.

---

## Loading market data

1. **Companies:** add them in the Django admin (`/admin/`), or import a company list.
   `script/test.py` is an importer that **deletes all companies** and recreates them from an Excel
   file. Edit its file path first and use it with care.
2. **Prices:** open **Daily Data Scraper** (`/daily-data/`), choose *Daily* or *5 Minute* and press
   **Start Processing**. The progress bar shows each company as it is downloaded.
   - Only one download runs at a time, because Yahoo Finance must be called sequentially.
   - Symbols that fail, or for which Yahoo has no data, are listed in `logs/fetch_errors/<job>_<time>.log`.
     The page links to the file.
   - Yahoo keeps only about 60 days of 5-minute data, so run the 5-minute download regularly.
3. **Analysis:** press **Refresh** on `/analysis/` to recompute the low / high statistics.
4. **Prediction models (optional):**
   ```bash
   python manage.py train_models --company TCS.NS     # omit --company to train all
   python manage.py test_predictions --company TCS.NS
   ```

---

## Research scripts

Standalone back-tests on the stored 5-minute data. Run them from the project folder; each one
writes an Excel report to `script/output/`. They use only the database (no Yahoo calls), and the
heavier ones run stocks in parallel processes.

| Script | Studies |
|---|---|
| `script/pdh_pdl_backtest.py` | Previous-day high / low **breakout** with a confirmation candle |
| `script/pdh_pdl_reversal_research.py` | Previous-day high / low **reversal** (failed breakout): signal time, breakout depth, volume, Nifty direction, plus an in-sample / out-of-sample filter test |
| `script/intraday_strategies_research.py` | The six Paper Trading strategies: targets, money-flow filter, time of day, long vs short, charges, walk-forward check of the page's *Qualified* label |
| `script/combined_strategy_research.py` | All of the above on one basis (R after charges, same liquid stocks), compared **by industry and sector** |

```bash
python script/combined_strategy_research.py                 # all stocks
python script/intraday_strategies_research.py --symbols TCS.NS,INFY.NS --workers 2
python script/pdh_pdl_reversal_research.py --help           # every option
```

Results are expressed in **R**: profit or loss divided by the risk taken (entry to stop), after
estimated intraday charges. A run over ~1,100 liquid NSE stocks (Dec 2025 – Oct 2026) found
**no strategy profitable after charges** in its default form. Gross results are close to zero, and the
charges on tight intraday stops turn them negative. The reports show where each strategy comes
closest and which industries deserve further testing.

---

## Configuration

Optional settings in `sethu_project/settings.py`:

| Setting | Default | Purpose |
|---|---|---|
| `PAPER_INTRADAY_SQUAREOFF_TIME` | `"15:20"` | Time live intraday positions are closed |
| `PAPER_LIVE_POLL_SECONDS` | `15` | How often the paper-trading page refreshes live data |
| `PAPER_LIVE_CACHE_SECONDS` | `10` | Server-side cache of live Yahoo candles |
| `PAPER_LIVE_MAX_STALENESS_MINUTES` | `20` | Live data older than this disables live orders |
| `ENABLE_MARKET_CONTEXT` | `False` | Add NIFTY / BANKNIFTY / India VIX features to predictions |
| `ML_MODELS_DIR` | `ml_models/` | Where trained prediction models are stored |

Before deploying, set your own `SECRET_KEY`, set `DEBUG = False`, and configure `ALLOWED_HOSTS`.

---

## Running the tests

```bash
python manage.py test paper_trading
```

---

## Known limitations

- **Background jobs** (data download, analysis refresh) keep their progress in Django's local-memory
  cache, which works with a single server process such as `runserver`. A multi-process deployment
  needs a shared cache such as Redis.
- **Yahoo Finance** reports zero volume for the 09:15 candle on most days, and offers only ~60 days of
  5-minute history.
- `TIME_ZONE` is `UTC`; the market logic converts to IST (Asia/Kolkata) where it matters.

---

## Author

**Sethumathavan**: [GitHub @Sethumathavan2001](https://github.com/Sethumathavan2001)
