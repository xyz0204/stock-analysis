#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oklo_auto_score_pipeline_v2.py
---------------------------------
End-to-end pipeline:
1) Slice a given ET time window into fixed intervals (e.g., 5 mins)
2) For each slice, call us_stock_data_to_csv.py to produce JSON
3) Combine slices and call strength_score_cli_v3.py to score & visualize

Examples:
  python oklo_auto_score_pipeline_v2.py \
    --tickers OKLO \
    --date 2025-10-29 \
    --start 09:35 --end 10:35 --interval 5 \
    --threshold 144.5 \
    --outdir data/out --xlsx

Multiple tickers + combined:
  python oklo_auto_score_pipeline_v2.py \
    --tickers OKLO LEU \
    --date 2025-10-29 \
    --start 09:30 --end 10:30 --interval 5 \
    --outdir out --xlsx --combined
"""
import argparse, os, sys, subprocess
from datetime import datetime, timedelta
import pytz

US_EASTERN = pytz.timezone("US/Eastern")

def parse_hhmm(s: str):
    try:
        h, m = s.split(":")
        return int(h), int(m)
    except Exception:
        raise argparse.ArgumentTypeError("Time must be HH:MM (e.g., 09:35)")

def make_time(date_et: str, hhmm: str):
    h, m = parse_hhmm(hhmm)
    day = datetime.strptime(date_et, "%Y-%m-%d").date()
    return US_EASTERN.localize(datetime(year=day.year, month=day.month, day=day.day, hour=h, minute=m))

def _minutes_between(date_et: str, start_hhmm: str, end_hhmm: str) -> int:
    t0 = make_time(date_et, start_hhmm)
    t1 = make_time(date_et, end_hhmm)
    if t1 < t0:
        t0, t1 = t1, t0
    delta = t1 - t0
    return int(delta.total_seconds() // 60)

def iter_slices(date_et: str, start_hhmm: str, end_hhmm: str, interval_min: int):
    t0 = make_time(date_et, start_hhmm)
    t1 = make_time(date_et, end_hhmm)
    if t1 <= t0:
        t0, t1 = t1, t0
    cur = t0
    out = []
    while cur < t1:
        nxt = cur + timedelta(minutes=interval_min)
        if nxt > t1:
            nxt = t1
        if nxt <= cur:
            # Defensive: skip zero/negative slice
            break
        out.append((cur.strftime("%H:%M"), nxt.strftime("%H:%M")))
        cur = nxt
    return out

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="End-to-end slicing + scoring pipeline (v2)")
    ap.add_argument("--tickers", nargs="+", required=True, help="One or more tickers, e.g., OKLO LEU")
    ap.add_argument("--date", required=True, help="ET date YYYY-MM-DD")
    ap.add_argument("--start", required=True, help="ET start time HH:MM (e.g., 09:35)")
    ap.add_argument("--end", required=True, help="ET end time HH:MM (e.g., 10:35)")
    ap.add_argument("--interval", type=int, default=5, help="Interval minutes per slice (default 5)")
    ap.add_argument("--outdir", default="out", help="Output directory for JSON & Excel (default: out)")
    ap.add_argument("--script-dir", default=".", help="Directory where us_stock_data_to_csv.py and strength_score_cli_v3.py reside")
    ap.add_argument("--threshold", type=float, default=None, help="Breakout threshold price passed to scorer")
    ap.add_argument("--per5avg", type=float, default=None, help="Override per-5min average volume in scorer")
    ap.add_argument("--xlsx", action="store_true", help="Write Excel with charts (requires XlsxWriter, scorer will fallback to CSV)")
    ap.add_argument("--combined", action="store_true", help="Also produce a combined (all tickers) scored file")
    ap.add_argument("--sort-by-period", action="store_true", help="Sort combined rows by period in scorer")
    ap.add_argument("--strict-divisible", action="store_true", help="Require (end-start) minutes be divisible by --interval")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # ---- Validate basic time/interval ----
    if args.interval is None or int(args.interval) < 1:
        print("[ERROR] --interval must be >= 1 minute.", file=sys.stderr); sys.exit(2)

    try:
        total_min = _minutes_between(args.date, args.start, args.end)
    except Exception as e:
        print(f"[ERROR] Invalid date/start/end: {e}", file=sys.stderr); sys.exit(2)

    if total_min == 0:
        print("[ERROR] start and end are identical; no time range to slice.", file=sys.stderr); sys.exit(2)

    if args.strict_divisible and (total_min % args.interval != 0):
        print(f"[ERROR] strict-divisible enabled: total minutes {total_min} not divisible by interval {args.interval}.", file=sys.stderr)
        sys.exit(2)

    if (total_min % args.interval) != 0:
        rem = total_min % args.interval
        print(f"[INFO] total minutes = {total_min}, interval = {args.interval} -> remainder = {rem} minute(s). "
              f"The final slice will be shorter.", file=sys.stderr)

    # Resolve script paths
    script_dir = args.script_dir
    grabber = os.path.join(script_dir, "us_stock_data_to_csv.py")
    scorer = os.path.join(script_dir, "strength_score_cli_v3.py")

    if not os.path.exists(grabber):
        print(f"[ERROR] Cannot find us_stock_data_to_csv.py at {grabber}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(scorer):
        print(f"[ERROR] Cannot find strength_score_cli_v3.py at {scorer}", file=sys.stderr); sys.exit(2)

    slices = iter_slices(args.date, args.start, args.end, args.interval)
    if not slices:
        print("[ERROR] No slices generated. Check start/end/interval.", file=sys.stderr); sys.exit(2)

    all_files_by_ticker = {}
    for ticker in args.tickers:
        json_files = []
        for (st, en) in slices:
            # Build JSON filename
            base = f"{ticker}_{args.date}_{st.replace(':','')}_{en.replace(':','')}.json"
            out_json = os.path.join(args.outdir, base)
            # Call grabber
            cmd = [
                sys.executable, grabber,
                "--tickers", ticker,
                "--date", args.date,
                "--start-time", st, "--end-time", en,
                "--json", out_json, "--json-format", "pretty",
                "--once", "--period-only"
            ]
            print("[RUN]", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Grab failed for {ticker} {st}-{en}: {e}", file=sys.stderr)
            if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
                json_files.append(out_json)
        if not json_files:
            print(f"[WARN] No JSON produced for {ticker}.", file=sys.stderr)
        all_files_by_ticker[ticker] = json_files

    # Score per ticker
    produced = []
    for ticker, files in all_files_by_ticker.items():
        if not files:
            continue
        out_name = os.path.join(args.outdir, f"{ticker}_{args.date}_scored.{'xlsx' if args.xlsx else 'csv'}")
        cmd = [
            sys.executable, scorer,
            "-i", *files,
            "-f", "json"
        ]
        if args.threshold is not None:
            cmd += ["-t", str(args.threshold)]
        if args.per5avg is not None:
            cmd += ["--per5avg", str(args.per5avg)]
        if args.sort_by_period:
            cmd += ["--sort-by-period"]
        if args.xlsx:
            cmd += ["--xlsx"]
        cmd += ["-o", out_name]
        print("[RUN]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            produced.append(out_name)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Scoring failed for {ticker}: {e}", file=sys.stderr)

    # Combined file across tickers (if requested)
    if args.combined:
        combined_list = []
        for files in all_files_by_ticker.values():
            combined_list.extend(files)
        if combined_list:
            out_name = os.path.join(args.outdir, f"combined_{args.date}_scored.{'xlsx' if args.xlsx else 'csv'}")
            cmd = [
                sys.executable, scorer,
                "-i", *combined_list,
                "-f", "json"
            ]
            if args.threshold is not None:
                cmd += ["-t", str(args.threshold)]
            if args.per5avg is not None:
                cmd += ["--per5avg", str(args.per5avg)]
            if args.sort_by_period:
                cmd += ["--sort-by-period"]
            if args.xlsx:
                cmd += ["--xlsx"]
            cmd += ["-o", out_name]
            print("[RUN]", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                produced.append(out_name)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Combined scoring failed: {e}", file=sys.stderr)

    if produced:
        print("\nProduced files:")
        for p in produced:
            print(" -", p)
    else:
        print("[WARN] No scored files produced.", file=sys.stderr)

if __name__ == "__main__":
    main()
