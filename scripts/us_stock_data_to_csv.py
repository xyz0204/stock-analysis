# -*- coding: utf-8 -*-
"""
us_stock_data_to_csv.py

Fetch intraday US stock data and export to CSV (or Excel) with flexible time-window selection.
Requires: pandas, yfinance, openpyxl (if using --xlsx)

Examples:
  # 取指定日期（美东时区），常用：FOMC当日 9:30–16:00
  python us_stock_data_to_csv.py -t OKLO -i 5m --date 2025-10-28

  # 自定义开始/结束（本地默认时区 America/New_York），时间精确到分钟
  python us_stock_data_to_csv.py -t LEU -i 1m --start "2025-10-28 09:30" --end "2025-10-28 11:00"

  # 多只标的（逗号分隔），输出到指定目录，并导出为 xlsx
  python us_stock_data_to_csv.py -t OKLO,LEU -i 15m --date 2025-10-28 --output-dir ./exports --xlsx

Notes:
- yfinance 对不同 interval 有历史范围限制（例如 1m 通常仅支持近些天）。
- 本脚本默认使用 America/New_York 时区进行 --date 与 --start/--end 的解释。
"""

import argparse
from datetime import datetime, timedelta
import os
import sys

try:
    import pandas as pd
except Exception as e:
    print("❌ Missing dependency: pandas. Please install via `pip install pandas`.", file=sys.stderr)
    raise

try:
    import yfinance as yf
except Exception as e:
    print("❌ Missing dependency: yfinance. Please install via `pip install yfinance`.", file=sys.stderr)
    raise

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    # For Python <3.9, fall back to pytz if available
    try:
        import pytz
        class ZoneInfo:
            def __init__(self, name): self.tz = pytz.timezone(name)
            def __call__(self, *a, **k): return self.tz
    except Exception:
        ZoneInfo = None

DEFAULT_TZ = "America/New_York"

def parse_args():
    p = argparse.ArgumentParser(description="Download US stock intraday data to CSV (or XLSX).")
    p.add_argument("--ticker", "-t", required=True,
                   help="Ticker symbol(s), e.g., OKLO or OKLO,LEU (comma-separated).")
    p.add_argument("--interval", "-i", default="5m",
                   help="Interval for data (e.g., 1m,2m,5m,15m,30m,60m,90m,1h,1d). Default: 5m")
    p.add_argument("--date", "-d", default=None,
                   help="Target date (YYYY-MM-DD) in America/New_York by default. Will use 09:30–16:00 window.")
    p.add_argument("--start", default=None,
                   help='Start datetime "YYYY-MM-DD HH:MM" (interpreted in --tz).')
    p.add_argument("--end", default=None,
                   help='End datetime "YYYY-MM-DD HH:MM" (interpreted in --tz).')
    p.add_argument("--tz", default=DEFAULT_TZ,
                   help=f"Timezone for --date / --start / --end (default: {DEFAULT_TZ})")
    p.add_argument("--prepost", action="store_true",
                   help="Include pre/post market data if available (yfinance prepost=True).")
    p.add_argument("--output-dir", default=".",
                   help="Directory to save output files. Default: current directory.")
    p.add_argument("--xlsx", action="store_true",
                   help="Export Excel (.xlsx) instead of CSV.")
    p.add_argument("--rename-cols", action="store_true",
                   help="Rename columns to open,high,low,close,volume (remove multi-index).")
    p.add_argument("--silent", action="store_true",
                   help="Reduce console output.")
    return p.parse_args()

def ensure_tz(dt: datetime, tzname: str) -> datetime:
    if ZoneInfo is None:
        # fallback naive (not recommended)
        return dt
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=ZoneInfo(tzname))

def build_time_window(args):
    tzname = args.tz or DEFAULT_TZ
    # Priority: --start/--end > --date > None
    if args.start or args.end:
        if not args.start or not args.end:
            raise ValueError("When using --start/--end, both must be provided.")
        start_local = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        end_local = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
        start_local = ensure_tz(start_local, tzname)
        end_local = ensure_tz(end_local, tzname)
        return start_local, end_local, f"{start_local:%Y%m%d_%H%M}-{end_local:%H%M}"
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
        start_local = ensure_tz(d.replace(hour=9, minute=30), tzname)
        end_local = ensure_tz(d.replace(hour=16, minute=0), tzname)
        return start_local, end_local, f"{d:%Y-%m-%d}"
    else:
        # No explicit window -> use period="1d"
        return None, None, "latest"

def download_one(ticker: str, interval: str, start_local, end_local, prepost: bool, silent: bool):
    if start_local is not None and end_local is not None:
        if not silent:
            print(f"↘ Downloading {ticker} {interval} {start_local} -> {end_local} (prepost={prepost})")
        df = yf.download(
            ticker,
            interval=interval,
            start=start_local,
            end=end_local,
            prepost=prepost,
            auto_adjust=False,
            progress=not silent
        )
    else:
        if not silent:
            print(f"↘ Downloading {ticker} period=1d interval={interval} (prepost={prepost})")
        df = yf.download(
            ticker,
            period="1d",
            interval=interval,
            prepost=prepost,
            auto_adjust=False,
            progress=not silent
        )
    return df

def normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    # yfinance often returns multiindex columns like ('Close','OKLO')
    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(axis=1)
        # flatten: use first level (ticker) if present
        if df.columns.nlevels >= 2:
            # pick the first column set for the ticker (Close/Open/High/Low/Volume)
            flat = {}
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    flat[col[-1]] = df[col]
                else:
                    flat[str(col)] = df[col]
            df = pd.DataFrame(flat, index=df.index)
    # harmonize names
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    # common aliases
    aliases = {"adj close": "adj_close"}
    df = df.rename(columns=aliases)
    return df

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    start_local, end_local, tag = build_time_window(args)

    for tk in tickers:
        df = download_one(tk, args.interval, start_local, end_local, args.prepost, args.silent)
        if df is None or df.empty:
            print(f"⚠ No data returned for {tk}. Check interval/date window/history limits.", file=sys.stderr)
            continue
        if args.rename_cols:
            df = normalize_columns(df)

        # build output path
        base = f"{tk}_{tag}_{args.interval}"
        out_path = os.path.join(args.output_dir, base + (".xlsx" if args.xlsx else ".csv"))

        if args.xlsx:
            try:
                df.to_excel(out_path, index=True)
            except Exception as e:
                print("❌ Writing .xlsx failed. Install openpyxl: `pip install openpyxl`", file=sys.stderr)
                raise
        else:
            df.to_csv(out_path, index=True, encoding="utf-8-sig")

        if not args.silent:
            print(f"✅ Saved -> {out_path}")
            # quick peek
            try:
                print(df.head(3))
            except Exception:
                pass

if __name__ == "__main__":
    main()
