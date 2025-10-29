#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
us_stock_data_to_csv.py  (JSON-enabled version)
-----------------------------------------------
Real-time-ish US stock monitor (yfinance, ~1m delayed).

Adds JSON output options on top of CSV:
- --json path/to/file.json
- --json-format ndjson|array|pretty   (default: ndjson)
  * ndjson : newline-delimited JSON (streaming-friendly; appends one object per cycle)
  * array  : JSON array; will append to existing array if possible, else create new
  * pretty : same as 'array' but pretty-printed (indent=2)

Other features unchanged:
- Session O/H/L/C/Volume
- Trailing-window O/H/L/C/Volume (last N minutes)
- 5-day average daily volume
- RSI(14) on intraday (fallback to daily)
- Period extraction with --start-time/--end-time (+ VWAP, range%, body%, volume ratio)
- Multi-ticker, CSV append, one-shot or loop

Examples:
  python us_stock_data_to_csv.py --tickers OKLO --json oklo.ndjson --json-format ndjson --period-only --start-time 09:35 --end-time 09:40
  python us_stock_data_to_csv.py --tickers OKLO LEU --csv live.csv --json live.ndjson
"""
import argparse, sys, time, json, os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np, pandas as pd, pytz

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance is required. Please install with `pip install yfinance`.", file=sys.stderr)
    sys.exit(1)

US_EASTERN = pytz.timezone("US/Eastern")

# ---------- Indicators ----------
def rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna()
    if len(s) < period + 1:
        return float('nan')
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])

# ---------- Data fetch ----------
def get_intraday_today(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize(US_EASTERN)
    else:
        df = df.tz_convert(US_EASTERN)
    return df

def get_daily_history(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=False)
    if df.empty:
        df = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=False)
    return df

def last_close_and_change(df_daily: pd.DataFrame) -> Tuple[float, float]:
    if len(df_daily) < 2:
        if len(df_daily) == 1:
            return float(df_daily["Close"].iloc[-1]), float('nan')
        return float('nan'), float('nan')
    last_close = float(df_daily["Close"].iloc[-1])
    prev_close = float(df_daily["Close"].iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else float('nan')
    return last_close, change_pct

# ---------- Aggregations ----------
def compute_session_ohlcv(df_intraday: pd.DataFrame) -> Dict[str, float]:
    if df_intraday.empty:
        return {k: float('nan') for k in ["session_open","session_high","session_low","session_close","session_volume"]}
    session_open = float(df_intraday["Open"].iloc[0])
    session_high = float(df_intraday["High"].max())
    session_low = float(df_intraday["Low"].min())
    session_close = float(df_intraday["Close"].iloc[-1])
    session_volume = int(df_intraday["Volume"].sum(skipna=True))
    return {
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "session_close": session_close,
        "session_volume": session_volume,
    }

def compute_trailing_window_ohlcv(df_intraday: pd.DataFrame, trailing_min: int) -> Dict[str, float]:
    if df_intraday.empty:
        return {k: float('nan') for k in ["tw_open","tw_high","tw_low","tw_close","tw_volume"]}
    end_ts = df_intraday.index[-1]
    start_ts = end_ts - pd.Timedelta(minutes=trailing_min)
    window = df_intraday.loc[df_intraday.index >= start_ts]
    if window.empty:
        return {k: float('nan') for k in ["tw_open","tw_high","tw_low","tw_close","tw_volume"]}
    tw_open = float(window["Open"].iloc[0])
    tw_high = float(window["High"].max())
    tw_low = float(window["Low"].min())
    tw_close = float(window["Close"].iloc[-1])
    tw_volume = int(window["Volume"].sum(skipna=True))
    return {
        "tw_open": tw_open,
        "tw_high": tw_high,
        "tw_low": tw_low,
        "tw_close": tw_close,
        "tw_volume": tw_volume,
    }

def compute_avg_vol_5d(df_daily: pd.DataFrame) -> float:
    if df_daily.empty:
        return float('nan')
    vols = df_daily["Volume"].dropna()
    if len(vols) >= 6:
        return float(vols.iloc[:-1].tail(5).mean())
    elif len(vols) >= 5:
        return float(vols.tail(5).mean())
    else:
        return float(vols.mean()) if len(vols) > 0 else float('nan')

# ---------- Period window ----------
def get_period_data(df_intraday: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df_intraday.empty:
        return df_intraday
    today = df_intraday.index[-1].astimezone(US_EASTERN).strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(f"{today} {start_time}", tz=US_EASTERN)
    end_ts = pd.Timestamp(f"{today} {end_time}", tz=US_EASTERN)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    sub = df_intraday.loc[(df_intraday.index >= start_ts) & (df_intraday.index <= end_ts)]
    return sub

def rsi_safe(series: pd.Series, period: int = 14) -> float:
    try:
        return rsi(series, period)
    except Exception:
        return float('nan')

def compute_period_metrics(df_period: pd.DataFrame) -> Dict[str, float]:
    if df_period.empty:
        return {k: float('nan') for k in [
            "period_open","period_high","period_low","period_close","period_volume","period_rsi14","period_minutes"
        ]}
    o = float(df_period["Open"].iloc[0])
    h = float(df_period["High"].max())
    l = float(df_period["Low"].min())
    c = float(df_period["Close"].iloc[-1])
    v = int(df_period["Volume"].sum(skipna=True))
    r = rsi_safe(df_period["Close"], 14)
    return {
        "period_open": o,
        "period_high": h,
        "period_low": l,
        "period_close": c,
        "period_volume": v,
        "period_rsi14": float(r),
        "period_minutes": int(len(df_period.index)),
    }

def compute_period_extras(df_period: pd.DataFrame,
                          period_open: float,
                          period_high: float,
                          period_low: float,
                          period_close: float) -> Dict[str, float]:
    if df_period.empty:
        return {"period_vwap": float('nan'),
                "period_range_pct": float('nan'),
                "period_body_pct": float('nan')}

    vol = df_period["Volume"].fillna(0)
    if vol.sum() > 0:
        typical_price = (df_period["High"] + df_period["Low"] + df_period["Close"]) / 3.0
        vwap = float((typical_price * vol).sum() / vol.sum())
    else:
        vwap = float('nan')

    if period_open and not np.isnan(period_open):
        if (period_high > period_low) and (period_open != 0):
            range_pct = (period_high - period_low) / period_open * 100.0
        else:
            range_pct = float('nan')
    else:
        range_pct = float('nan')

    denom = (period_high - period_low)
    body_pct = (abs(period_close - period_open) / denom * 100.0) if (denom > 0) else float('nan')

    return {
        "period_vwap": vwap,
        "period_range_pct": float(range_pct),
        "period_body_pct": float(body_pct),
    }

# ---------- Helpers ----------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

def ensure_columns_order(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = float('nan')
    return df[columns]

# ---------- Row build ----------
def build_row(ticker: str,
              trailing_min: int,
              start_time: Optional[str] = None,
              end_time: Optional[str] = None) -> Dict[str, float]:
    df_intraday = get_intraday_today(ticker)
    df_daily = get_daily_history(ticker)

    if not df_intraday.empty:
        last_price = float(df_intraday["Close"].iloc[-1])
        last_ts = df_intraday.index[-1]
    elif not df_daily.empty:
        last_price = float(df_daily["Close"].iloc[-1])
        last_ts = pd.Timestamp.now(tz=US_EASTERN)
    else:
        last_price, last_ts = float('nan'), pd.Timestamp.now(tz=US_EASTERN)

    _, change_pct = last_close_and_change(df_daily)
    rsi_val = rsi_safe(df_intraday["Close"], period=14) if not df_intraday.empty and len(df_intraday) >= 15 \
              else (rsi_safe(df_daily["Close"], period=14) if not df_daily.empty else float('nan'))

    session = compute_session_ohlcv(df_intraday)
    trailing = compute_trailing_window_ohlcv(df_intraday, trailing_min=trailing_min)
    avg5d_vol = compute_avg_vol_5d(df_daily)

    row: Dict[str, float] = {
        "timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "ticker": ticker,
        "last_price": safe_float(last_price),
        "pct_change": safe_float(change_pct),
        "rsi14": safe_float(rsi_val),
        **session,
        **trailing,
        "avg_vol_5d": safe_float(avg5d_vol),
    }

    if start_time and end_time:
        df_period = get_period_data(df_intraday, start_time, end_time)
        period = compute_period_metrics(df_period)
        row.update(period)
        row["period_start_et"] = start_time
        row["period_end_et"] = end_time

        extras = compute_period_extras(
            df_period,
            period_open=row["period_open"],
            period_high=row["period_high"],
            period_low=row["period_low"],
            period_close=row["period_close"],
        )
        row.update(extras)

        ses_vol = row.get("session_volume", float('nan'))
        if (isinstance(ses_vol, (int, float)) and ses_vol and not np.isnan(ses_vol) and ses_vol > 0):
            row["period_vol_ratio"] = float(row["period_volume"]) / float(ses_vol) * 100.0
        else:
            row["period_vol_ratio"] = float('nan')
    else:
        row.update({
            "period_open": float('nan'),
            "period_high": float('nan'),
            "period_low": float('nan'),
            "period_close": float('nan'),
            "period_volume": float('nan'),
            "period_rsi14": float('nan'),
            "period_minutes": float('nan'),
            "period_start_et": "",
            "period_end_et": "",
            "period_vwap": float('nan'),
            "period_range_pct": float('nan'),
            "period_vol_ratio": float('nan'),
            "period_body_pct": float('nan'),
        })
    return row

# ---------- CSV / JSON I/O ----------
def write_header_if_needed(csv_path: Optional[str], columns: List[str]) -> None:
    if not csv_path:
        return
    try:
        with open(csv_path, "r", encoding="utf-8") as _:
            return
    except FileNotFoundError:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False, encoding="utf-8")

def append_rows_csv(csv_path: Optional[str], rows: List[Dict[str, float]], columns: List[str]) -> None:
    if not csv_path:
        return
    df = pd.DataFrame(rows)
    df = ensure_columns_order(df, columns)
    df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")

def _load_json_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def append_rows_json(json_path: Optional[str], rows: List[Dict[str, float]], json_format: str = "ndjson") -> None:
    if not json_path:
        return
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    if json_format == "ndjson":
        with open(json_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        # array / pretty
        existing = _load_json_file(json_path)
        if isinstance(existing, list):
            existing.extend(rows)
            data = existing
        else:
            data = rows  # start fresh if file corrupt/non-list

        with open(json_path, "w", encoding="utf-8") as f:
            if json_format == "pretty":
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="US stock live monitor to CSV/JSON with optional ET period extraction.")
    p.add_argument("--tickers", nargs="+", required=True, help="One or more ticker symbols, e.g., OKLO LEU SMR AAPL")
    p.add_argument("--csv", default=None, help="Output CSV path (optional)")
    p.add_argument("--json", default=None, help="Output JSON file path (optional)")
    p.add_argument("--json-format", default="ndjson", choices=["ndjson","array","pretty"], help="JSON format (default: ndjson)")
    p.add_argument("--interval-sec", type=int, default=60, help="Polling interval seconds (default: 60)")
    p.add_argument("--trailing-min", type=int, default=15, help="Trailing window minutes for O/H/L/C/Vol (default: 15)")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p.add_argument("--start-time", type=str, help="ET start time for period extraction, e.g., 09:35")
    p.add_argument("--end-time", type=str, help="ET end time for period extraction, e.g., 09:40")
    p.add_argument("--period-only", action="store_true", help="Only compute the provided period metrics and exit")
    return p.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()

    columns: List[str] = [
        "timestamp","ticker","last_price","pct_change","rsi14",
        "session_open","session_high","session_low","session_close","session_volume",
        "tw_open","tw_high","tw_low","tw_close","tw_volume",
        "avg_vol_5d",
        "period_open","period_high","period_low","period_close","period_volume","period_rsi14","period_minutes",
        "period_start_et","period_end_et",
        "period_vwap","period_range_pct","period_vol_ratio","period_body_pct",
    ]
    # Only create CSV header when CSV is requested
    if args.csv:
        write_header_if_needed(args.csv, columns)

    def one_cycle():
        rows: List[Dict[str, float]] = []
        for t in args.tickers:
            try:
                row = build_row(t, args.trailing_min, args.start_time, args.end_time)
            except Exception as e:
                row = {
                    "timestamp": datetime.now(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "ticker": t,
                    "last_price": float('nan'),
                    "pct_change": float('nan'),
                    "rsi14": float('nan'),
                    "session_open": float('nan'),
                    "session_high": float('nan'),
                    "session_low": float('nan'),
                    "session_close": float('nan'),
                    "session_volume": float('nan'),
                    "tw_open": float('nan'),
                    "tw_high": float('nan'),
                    "tw_low": float('nan'),
                    "tw_close": float('nan'),
                    "tw_volume": float('nan'),
                    "avg_vol_5d": float('nan'),
                    "period_open": float('nan'),
                    "period_high": float('nan'),
                    "period_low": float('nan'),
                    "period_close": float('nan'),
                    "period_volume": float('nan'),
                    "period_rsi14": float('nan'),
                    "period_minutes": float('nan'),
                    "period_start_et": args.start_time or "",
                    "period_end_et": args.end_time or "",
                    "period_vwap": float('nan'),
                    "period_range_pct": float('nan'),
                    "period_vol_ratio": float('nan'),
                    "period_body_pct": float('nan'),
                }
                print(f"[WARN] Failed to fetch {t}: {e}", file=sys.stderr)
            rows.append(row)

        # CSV append
        if args.csv:
            append_rows_csv(args.csv, rows, columns)

        # JSON append
        if args.json:
            append_rows_json(args.json, rows, args.json_format)

        # stdout (unchanged)
        for r in rows:
            base = (f"{r['timestamp']}  {r['ticker']:>6}  {r['last_price']!s:>8}  Δ%={r['pct_change']!s:>6}  "
                    f"RSI14={r['rsi14']!s:>6}  Ses O/H/L/C={r['session_open']}/{r['session_high']}/"
                    f"{r['session_low']}/{r['session_close']}  TW{args.trailing_min} O/H/L/C="
                    f"{r['tw_open']}/{r['tw_high']}/{r['tw_low']}/{r['tw_close']}  Vol Ses={r['session_volume']} "
                    f"TW={r['tw_volume']} Avg5dVol={r['avg_vol_5d']}")
            if args.start_time and args.end_time:
                base += (f"  |  PERIOD[{r['period_start_et']}-{r['period_end_et']}] "
                         f"O/H/L/C={r['period_open']}/{r['period_high']}/{r['period_low']}/{r['period_close']} "
                         f"Vol={r['period_volume']} RSI={r['period_rsi14']} nmin={r['period_minutes']} "
                         f"VWAP={r['period_vwap']} Range%={r['period_range_pct']} "
                         f"Vol%Ses={r['period_vol_ratio']} Body%={r['period_body_pct']}")
            print(base)

    # period-only one shot
    if args.period_only and args.start_time and args.end_time:
        one_cycle()
        return

    if args.once:
        one_cycle()
        return

    try:
        while True:
            one_cycle()
            time.sleep(max(5, args.interval_sec))
    except KeyboardInterrupt:
        print("\nStopped by user. Bye.")

if __name__ == "__main__":
    main()
