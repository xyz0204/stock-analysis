# STOCK Data Analysis Toolkit

A lightweight toolkit for automated U.S. stock data collection and intraday candle strength analysis —  
designed for quantitative trading workflows focused on the nuclear energy and quantum computing sectors.

---

## 🚀 Features

- **Multi-provider data collection**  
  Supports both [Yahoo Finance](https://finance.yahoo.com) (`yfinance`) and [Finnhub](https://finnhub.io) APIs.
- **Flexible time windows**  
  Choose either a full trading day (`--date`) or custom `--start` / `--end` intervals (America/New_York by default).
- **Unified OHLCV schema**  
  Output always includes: `timestamp, open, high, low, close, volume`.
- **Multi-format export**  
  Supports `.csv`, `.xlsx`, and `.json`.
- **5-minute candle strength scoring**  
  Combines volume ratio, VWAP deviation, and breakout confirmation logic.

---

## 🧠 Example Usage

### 1. Collect intraday data

```bash
# Using Yahoo Finance (default)
python scripts/us_stock_data_to_csv_v3.py \
    -t AAPL -i 5m --date 2025-10-28 --provider yfinance

# Using Finnhub (real-time) with JSON export
python scripts/us_stock_data_to_csv_v3.py \
    -t AAPL -i 1m --date 2025-10-28 \
    --provider finnhub --token YOUR_API_KEY --json

