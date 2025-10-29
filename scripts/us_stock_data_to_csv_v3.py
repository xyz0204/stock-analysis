# -*- coding: utf-8 -*-
"""
us_stock_data_to_csv_v3.py
---------------------------------
Multi-provider (yfinance | finnhub) intraday downloader with unified output schema.
- Supports --date (09:30–16:00 America/New_York) or custom --start/--end.
- Outputs CSV (default), XLSX (--xlsx) or JSON (--json).
- Unified columns: timestamp, open, high, low, close, volume
- Optional: --rename-cols to enforce lower-case & flatten

Install:
  pip install pandas yfinance openpyxl finnhub-python pytz

Examples:
  # yfinance → CSV（默认）
  python us_stock_data_to_csv_v3.py -t OKLO -i 5m --date 2025-10-28 --provider yfinance

  # finnhub → JSON
  python us_stock_data_to_csv_v3.py -t OKLO -i 1m --date 2025-10-28 --provider finnhub --token YOUR_API_KEY --json

  # yfinance → Excel
  python us_stock_data_to_csv_v3.py -t LEU -i 15m --date 2025-10-28 --provider yfinance --xlsx
"""

import argparse
from datetime import datetime
import os
import sys
import json

import pandas as pd

# Optional imports by provider
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import finnhub
except Exception:
    finnhub = None

# Timezone handling
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    try:
        import pytz
        class ZoneInfo:
            def __init__(self, name): self.tz = pytz.timezone(name)
            def __call__(self, *a, **k): return self.tz
    except Exception:
        ZoneInfo = None

DEFAULT_TZ = "America/New_York"

def parse_args():
    p = argparse.ArgumentParser(description="Download US intraday data with multi-provider (yfinance|finnhub).")
    p.add_argument("--provider", default="yfinance", choices=["yfinance","finnhub"], help="Data provider")
    p.add_argument("--token", default=None, help="API key (required for finnhub)")
    p.add_argument("--ticker", "-t", required=True, help="Ticker(s), comma-separated, e.g., OKLO,LEU")
    p.add_argument("--interval", "-i", default="5m", help="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d")
    p.add_argument("--date", "-d", default=None, help="Target date (YYYY-MM-DD), 09:30–16:00 (America/New_York)")
    p.add_argument("--start", default=None, help='Start "YYYY-MM-DD HH:MM" (interpreted in --tz)')
    p.add_argument("--end", default=None, help='End "YYYY-MM-DD HH:MM" (interpreted in --tz)')
    p.add_argument("--tz", default=DEFAULT_TZ, help=f"Timezone for --date/--start/--end (default: {DEFAULT_TZ})")
    p.add_argument("--prepost", action="store_true", help="Include pre/post market when provider=yfinance")
    p.add_argument("--output-dir", default=".", help="Output directory")
    p.add_argument("--xlsx", action="store_true", help="Write .xlsx instead of .csv")
    p.add_argument("--json", action="store_true", help="Write .json instead of .csv/.xlsx")
    p.add_argument("--rename-cols", action="store_true", help="Lower-case & flatten column names")
    p.add_argument("--silent", action="store_true", help="Reduce console output")
    return p.parse_args()

def ensure_tz(dt: datetime, tzname: str) -> datetime:
    if ZoneInfo is None:
        return dt  # naive fallback
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=ZoneInfo(tzname))

def build_time_window(args):
    tzname = args.tz or DEFAULT_TZ
    if args.start or args.end:
        if not args.start or not args.end:
            raise ValueError("When using --start/--end, both must be provided.")
        start_local = ensure_tz(datetime.strptime(args.start, "%Y-%m-%d %H:%M"), tzname)
        end_local = ensure_tz(datetime.strptime(args.end, "%Y-%m-%d %H:%M"), tzname)
        tag = f"{start_local:%Y%m%d_%H%M}-{end_local:%H%M}"
        return start_local, end_local, tag
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
        start_local = ensure_tz(d.replace(hour=9, minute=30), tzname)
        end_local = ensure_tz(d.replace(hour=16, minute=0), tzname)
        tag = f"{d:%Y-%m-%d}"
        return start_local, end_local, tag
    else:
        return None, None, "latest"

def normalize_df_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if str(c)!='']) for col in df.columns]
    # lower-case
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # map common names
    rename_map = {
        "adj close": "adj_close",
        "close_close": "close",
        "open_open": "open",
        "high_high": "high",
        "low_low": "low",
        "volume_volume": "volume"
    }
    df = df.rename(columns=rename_map)
    return df

def provider_yfinance(ticker: str, interval: str, start_local, end_local, prepost: bool, silent: bool) -> "pd.DataFrame":
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")
    if start_local is not None:
        if not silent:
            print(f"[yfinance] {ticker} {interval} {start_local} -> {end_local}, prepost={prepost}")
        df = yf.download(ticker, interval=interval, start=start_local, end=end_local, prepost=prepost, auto_adjust=False, progress=not silent)
    else:
        if not silent:
            print(f"[yfinance] {ticker} period=1d interval={interval}, prepost={prepost}")
        df = yf.download(ticker, period="1d", interval=interval, prepost=prepost, auto_adjust=False, progress=not silent)
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if "Open" in df.columns:
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df = normalize_df_columns(df)
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing column from yfinance data: {c}")
    df = df[["open","high","low","close","volume"]]
    df.insert(0, "timestamp", df.index.tz_localize(None))
    return df

def provider_finnhub_fetch(client, ticker: str, interval: str, start_local, end_local, silent: bool) -> "pd.DataFrame":
    res_map = {"1m":"1","2m":"1","5m":"5","15m":"15","30m":"30","60m":"60","90m":"60","1h":"60","1d":"D"}
    if interval not in res_map:
        raise ValueError(f"Interval {interval} not supported by finnhub. Use one of: {', '.join(res_map.keys())}")
    reso = res_map[interval]
    if start_local is None or end_local is None:
        raise ValueError("finnhub requires explicit start/end or --date window.")
    start_ts = int(start_local.timestamp())
    end_ts = int(end_local.timestamp())
    if not silent:
        print(f"[finnhub] {ticker} reso={reso} {start_local} -> {end_local}")
    data = client.stock_candles(ticker, reso, start_ts, end_ts)
    if not data or ('s' in data and data['s'] != 'ok'):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    rename = {"o":"open","h":"high","l":"low","c":"close","v":"volume","t":"timestamp"}
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df[["timestamp","open","high","low","close","volume"]]
    return df

def provider_finnhub_main(token: str, ticker: str, interval: str, start_local, end_local, silent: bool) -> "pd.DataFrame":
    if finnhub is None:
        raise RuntimeError("finnhub not installed. pip install finnhub-python")
    if not token:
        raise ValueError("Provider 'finnhub' requires --token")
    client = finnhub.Client(api_key=token)
    return provider_finnhub_fetch(client, ticker, interval, start_local, end_local, silent)

def save_output(df: "pd.DataFrame", out_base: str, to_json: bool, to_xlsx: bool, silent: bool):
    if to_json:
        out_path = out_base + ".json"
        # ensure timestamp is ISO string
        df_to_json = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_to_json["timestamp"]):
            df_to_json["timestamp"] = df_to_json["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_to_json.to_json(out_path, orient="records", indent=2, force_ascii=False)
        if not silent:
            print(f"✅ Saved JSON -> {out_path}")
        return
    if to_xlsx:
        out_path = out_base + ".xlsx"
        try:
            df.to_excel(out_path, index=False)
        except Exception as e:
            print("❌ Writing .xlsx failed. Install openpyxl: `pip install openpyxl`", file=sys.stderr)
            raise
        if not silent:
            print(f"✅ Saved XLSX -> {out_path}")
        return
    # default CSV
    out_path = out_base + ".csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not silent:
        print(f"✅ Saved CSV -> {out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    # build time window tag
    start_local = end_local = None
    tag = "latest"
    try:
        start_local, end_local, tag = build_time_window(args)
    except Exception as e:
        print(f"❌ Time window error: {e}", file=sys.stderr)
        sys.exit(2)

    for tk in tickers:
        if args.provider == "finnhub":
            try:
                df = provider_finnhub_main(args.token, tk, args.interval, start_local, end_local, args.silent)
            except Exception as e:
                print(f"❌ finnhub error for {tk}: {e}", file=sys.stderr)
                continue
        else:
            try:
                df = provider_yfinance(tk, args.interval, start_local, end_local, args.prepost, args.silent)
            except Exception as e:
                print(f"❌ yfinance error for {tk}: {e}", file=sys.stderr)
                continue

        if df is None or df.empty:
            print(f"⚠ No data for {tk} (provider={args.provider}).", file=sys.stderr)
            continue

        if args.rename_cols:
            df = df.rename(columns={c: c.lower() for c in df.columns})

        base = os.path.join(args.output_dir, f"{tk}_{args.provider}_{tag}_{args.interval}")
        save_output(df, base, to_json=args.json, to_xlsx=args.xlsx, silent=args.silent)

        if not args.silent:
            try:
                print(df.head(3).to_string(index=False))
            except Exception:
                pass

if __name__ == "__main__":
    main()
