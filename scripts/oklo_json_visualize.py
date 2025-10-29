#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oklo_json_visualize.py
----------------------
Visualize minute-level OHLC/Close, VWAP, Volume, and RSI(14) for a given ticker and time window,
using your JSON output from us_stock_data_to_csv.py to guide the window, and fetching minute bars via yfinance.

Charts:
  1) Price (Close) with VWAP overlay (one figure)
  2) Volume bars (one figure)
  3) RSI(14) line (one figure)

Requirements:
  pip install yfinance pandas numpy matplotlib pytz

Notes:
- This script will fetch minute bars for the given date from yfinance (1-minute, ~1m delayed).
- If --date/--period are not provided, it will auto-detect the latest record for the ticker in the JSON.
- JSON can be NDJSON (one json per line) or an array. We'll auto-detect.
"""

import argparse, json, sys
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance is required. Please install with `pip install yfinance`.", file=sys.stderr)
    sys.exit(1)

US_EASTERN = pytz.timezone("US/Eastern")

def load_json_records(path: str) -> List[Dict[str, Any]]:
    # Try NDJSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # if first/last char look like array, parse as array
        if lines and lines[0].startswith('['):
            return json.load(open(path, "r", encoding="utf-8"))
        records = [json.loads(ln) for ln in lines]
        return records
    except Exception:
        # Fallback to array
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read JSON file: {e}", file=sys.stderr)
            return []

def pick_record(records: List[Dict[str, Any]], ticker: str,
                dt_str: Optional[str], period_str: Optional[str]) -> Optional[Dict[str, Any]]:
    # Filter by ticker
    recs = [r for r in records if r.get("ticker") == ticker]
    if not recs:
        return None

    if dt_str and period_str:
        # Match exact date+period
        pstart, pend = period_str.split("-")
        for r in reversed(recs):
            ts = r.get("timestamp", "")
            # parse date part from timestamp "YYYY-MM-DD HH:MM:SS EDT"
            d = ts.split(" ")[0] if ts else ""
            if d == dt_str and r.get("period_start_et")==pstart and r.get("period_end_et")==pend:
                return r
    elif dt_str:
        for r in reversed(recs):
            d = r.get("timestamp","").split(" ")[0]
            if d == dt_str:
                return r
    # Fallback: latest record
    return recs[-1]

def fetch_minute_bars(ticker: str, session_date: str) -> pd.DataFrame:
    """
    Fetch 1-minute bars for `session_date` (YYYY-MM-DD, ET) using yfinance.
    yfinance doesn't accept exact date easily, so we pull 1 day and rely on local filtering.
    """
    df = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize(US_EASTERN)
    else:
        df = df.tz_convert(US_EASTERN)
    # Keep only the target date
    df = df.loc[df.index.strftime("%Y-%m-%d") == session_date]
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(index=series.index, dtype=float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    # reindex to original to align
    rsi_series = rsi_series.reindex(series.index).astype(float)
    return rsi_series

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0).astype(float)
    cum_val = (typical * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)
    vwap = cum_val / cum_vol
    return vwap

def plot_price_with_vwap(df: pd.DataFrame, title: str):
    plt.figure()
    df["Close"].plot()
    vwap = compute_vwap(df)
    vwap.plot()
    plt.title(title)
    plt.xlabel("Time (ET)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

def plot_volume(df: pd.DataFrame, title: str):
    plt.figure()
    plt.bar(df.index, df["Volume"].astype(float).values, width=0.0006)  # narrow bars
    plt.title(title)
    plt.xlabel("Time (ET)")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()

def plot_rsi(df: pd.DataFrame, title: str):
    plt.figure()
    r = rsi(df["Close"], 14)
    r.plot()
    plt.axhline(70)
    plt.axhline(30)
    plt.title(title)
    plt.xlabel("Time (ET)")
    plt.ylabel("RSI(14)")
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Visualize minute bars (Price+VWAP, Volume, RSI) for a ticker and time window.")
    ap.add_argument("--json", required=True, help="Path to the JSON file (ndjson or array)")
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., OKLO")
    ap.add_argument("--date", help="Session date in ET, YYYY-MM-DD; if omitted, auto from latest JSON record")
    ap.add_argument("--period", help="Time window like HH:MM-HH:MM in ET; if omitted, auto from latest JSON record")
    args = ap.parse_args()

    records = load_json_records(args.json)
    if not records:
        print("No records loaded from JSON.", file=sys.stderr)
        sys.exit(2)

    rec = pick_record(records, args.ticker, args.date, args.period)
    if not rec:
        print("No matching record found. Check --ticker/--date/--period.", file=sys.stderr)
        sys.exit(3)

    # derive date and period from record if not provided
    ts = rec.get("timestamp","")
    dt = (args.date or (ts.split(" ")[0] if ts else None))
    pstart = rec.get("period_start_et") if not args.period else args.period.split("-")[0]
    pend = rec.get("period_end_et") if not args.period else args.period.split("-")[1]

    if not dt or not pstart or not pend:
        print("Cannot determine date/period. Provide --date and --period explicitly.", file=sys.stderr)
        sys.exit(4)

    # fetch minute bars and slice
    df = fetch_minute_bars(args.ticker, dt)
    if df.empty:
        print("No minute bars fetched for given date.", file=sys.stderr)
        sys.exit(5)

    start_ts = pd.Timestamp(f"{dt} {pstart}", tz=US_EASTERN)
    end_ts = pd.Timestamp(f"{dt} {pend}", tz=US_EASTERN)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    window = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    if window.empty:
        print("No data in specified window.", file=sys.stderr)
        sys.exit(6)

    # Draw charts
    plot_price_with_vwap(window, f"{args.ticker} {dt} {pstart}-{pend}  Close + VWAP")
    plot_volume(window, f"{args.ticker} {dt} {pstart}-{pend}  Volume")
    plot_rsi(window, f"{args.ticker} {dt} {pstart}-{pend}  RSI(14)")

if __name__ == "__main__":
    main()
