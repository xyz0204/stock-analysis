# -*- coding: utf-8 -*-
"""
strength_score_cli_v2.py — enhanced 5-minute strength scoring (single-row viz improved)

Changelog since v2:
- If only 1 row of data, Excel export will insert:
  * A scatter chart with a single marker (score) and data label
  * A bar chart for vol_factor
  * A KPI box (Score/Bias/VWAP delta/Volume factor) for quick glance
"""

import json
import math
import argparse
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import os
import sys

# -------- Helpers --------
def compute_per5min_avg_from_5d(avg_vol_5d: Optional[float]) -> Optional[float]:
    try:
        if avg_vol_5d is None: 
            return None
        return float(avg_vol_5d) / 78.0  # 6.5h * 60 / 5 = 78 bars
    except Exception:
        return None

def classify_candle(open_p: float, high: float, low: float, close: float) -> Dict[str, float]:
    rng = max(high - low, 1e-9)
    body = abs(close - open_p)
    upper = max(high - max(open_p, close), 0.0)
    lower = max(min(open_p, close) - low, 0.0)
    body_pct = (body / rng) * 100.0
    upper_pct = (upper / rng) * 100.0
    lower_pct = (lower / rng) * 100.0
    is_bull = close >= open_p
    if body_pct < 20 and upper_pct > 50 and not is_bull:
        pattern = "长上影阴线"
    elif body_pct < 20 and lower_pct > 50 and is_bull:
        pattern = "长下影阳线"
    elif is_bull and body_pct >= 40:
        pattern = "中大阳线"
    elif (not is_bull) and body_pct >= 40:
        pattern = "中大阴线"
    else:
        pattern = "十字/小实体K"
    return {
        "range_pct": (rng / max(open_p, 1e-9)) * 100.0,
        "body_pct": body_pct,
        "upper_shadow_pct": upper_pct,
        "lower_shadow_pct": lower_pct,
        "is_bull": 1 if is_bull else 0,
        "pattern_hint": pattern
    }

def strength_score_5min(
    bar: Dict,
    prior_bars: Optional[List[Dict]] = None,
    per5min_avg_vol_override: Optional[float] = None,
    breakout_threshold: Optional[float] = None,
    trend_window: int = 3
) -> Dict:
    # ---- Extract fields (fallbacks included for compatibility) ----
    open_p = float(bar.get("period_open", bar.get("session_open", bar.get("open", 0.0))))
    high = float(bar.get("period_high", bar.get("session_high", bar.get("high", open_p))))
    low = float(bar.get("period_low", bar.get("session_low", bar.get("low", open_p))))
    close = float(bar.get("period_close", bar.get("session_close", bar.get("close", open_p))))
    volume = float(bar.get("period_volume", bar.get("volume", 0.0)))
    vwap = float(bar.get("period_vwap", bar.get("session_vwap", bar.get("vwap", close))))
    avg_vol_5d = bar.get("avg_vol_5d", None)

    # ---- Per-5min avg volume ----
    if per5min_avg_vol_override is not None:
        per5min_avg = per5min_avg_vol_override
    else:
        per5min_avg = compute_per5min_avg_from_5d(avg_vol_5d)
    if not per5min_avg or per5min_avg <= 0:
        per5min_avg = max(volume, 1.0)  # avoid /0; treat current vol as baseline

    vol_factor = volume / per5min_avg

    # ---- Candle classification ----
    candle = classify_candle(open_p, high, low, close)

    # ---- Base score ----
    score = 5.0
    reasons: List[str] = []

    # ---- Volume contribution (smooth mapping; bounded [-2, +2]) ----
    vol_delta = np.clip((vol_factor - 1.0) * 1.2, -2.0, 2.0)
    score += float(vol_delta)
    if vol_factor >= 3.0:
        reasons.append(f"放量{vol_factor:.1f}x（极强）")
    elif vol_factor >= 1.5:
        reasons.append(f"放量{vol_factor:.1f}x")
    elif vol_factor <= 0.4:
        reasons.append(f"严重缩量{vol_factor:.2f}x")
    elif vol_factor <= 0.7:
        reasons.append(f"缩量{vol_factor:.2f}x")
    else:
        reasons.append(f"量能接近均值{vol_factor:.2f}x")

    # ---- VWAP proximity ----
    delta_vwap_pct = ((close - vwap) / max(vwap, 1e-9)) * 100.0
    if delta_vwap_pct >= 1.0:
        score += 2.0; reasons.append(f"收盘高于VWAP {delta_vwap_pct:.2f}%（强）")
    elif delta_vwap_pct >= 0.4:
        score += 1.0; reasons.append(f"收盘略高于VWAP {delta_vwap_pct:.2f}%")
    elif delta_vwap_pct <= -1.0:
        score -= 2.0; reasons.append(f"收盘低于VWAP {delta_vwap_pct:.2f}%（弱）")
    elif delta_vwap_pct <= -0.4:
        score -= 1.0; reasons.append(f"收盘略低于VWAP {delta_vwap_pct:.2f}%")
    else:
        reasons.append(f"收盘接近VWAP {delta_vwap_pct:.2f}%")

    # ---- Candle body/shadows ----
    body_pct = candle["body_pct"]; upper_pct = candle["upper_shadow_pct"]; lower_pct = candle["lower_shadow_pct"]; is_bull = bool(candle["is_bull"])
    if is_bull and body_pct >= 40:
        score += 1.0; reasons.append("中大阳线（实体≥40%）")
    if (not is_bull) and body_pct >= 40:
        score -= 1.0; reasons.append("中大阴线（实体≥40%）")
    if upper_pct >= 65:
        score -= 2.0; reasons.append("极长上影（≥65%）")
    elif upper_pct >= 50:
        score -= 1.0; reasons.append("长上影（≥50%）")
    if lower_pct >= 50 and is_bull:
        score += 1.0; reasons.append("长下影阳线（承接）")

    # ---- Trend consistency across last N bars (skip if insufficient history) ----
    if prior_bars and trend_window and trend_window > 1:
        k = min(len(prior_bars), trend_window)
        if k >= 2:
            ups = sum(float(b.get("period_close", b.get("session_close", b.get("close", 0.0)))) >=
                      float(b.get("period_open", b.get("session_open", b.get("open", 0.0))))
                      for b in prior_bars[-k:])
            trend_up = ups >= (k//2 + 1)
            if trend_up and is_bull:
                score += 0.5; reasons.append(f"短线趋势一致（近{k}根阳线≥{k//2+1}）")
            downs = k - ups
            trend_down = downs >= (k//2 + 1)
            if trend_down and (not is_bull):
                score -= 0.5; reasons.append(f"短线趋势一致（近{k}根阴线≥{k//2+1}）")
    # If no/insufficient history: no bonus/penalty.

    # ---- Breakout logic ----
    breakout = False
    if breakout_threshold is not None:
        if close > breakout_threshold and vol_factor >= 1.5:
            sustained = False
            if prior_bars and len(prior_bars) >= 1:
                prev_close = float(prior_bars[-1].get("period_close", prior_bars[-1].get("session_close", prior_bars[-1].get("close", close))))
                sustained = (prev_close > breakout_threshold)
            if sustained:
                score += 2.0; breakout = True; reasons.append(f"突破并连续站上 {breakout_threshold}（确认）")
            else:
                score += 1.0; breakout = True; reasons.append(f"放量突破 {breakout_threshold}")
        elif high > breakout_threshold and close <= breakout_threshold:
            score -= 1.0; reasons.append(f"冲高未站稳 {breakout_threshold}（疑似假突破）")

    # ---- RSI assistance if present ----
    rsi_val = bar.get("rsi14", None)
    try:
        rsi_val = float(rsi_val) if rsi_val is not None and str(rsi_val) != "nan" else None
    except Exception:
        rsi_val = None
    if rsi_val is not None:
        if rsi_val >= 70:
            score -= 0.5; reasons.append("RSI高位（≥70）")
        elif rsi_val <= 30:
            score += 0.5; reasons.append("RSI低位（≤30）")

    # ---- Clamp and bias ----
    score = max(0.0, min(10.0, float(score)))
    if score >= 7.5: bias = "强势"
    elif score >= 5.5: bias = "偏强"
    elif score >= 4.5: bias = "中性"
    elif score >= 3.5: bias = "偏弱"
    else: bias = "弱势"

    # ---- Period label ----
    period = bar.get("period_start_et") or bar.get("timestamp") or bar.get("time") or ""

    return {
        "source_file": bar.get("__source_file__", ""),
        "ticker": bar.get("ticker", ""),
        "period": period,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume, "per5min_avg": per5min_avg, "vol_factor": vol_factor,
        "vwap": vwap, "delta_vwap_pct": ((close - vwap) / max(vwap, 1e-9)) * 100.0,
        "candle_pattern": candle["pattern_hint"],
        "body_pct": body_pct, "upper_shadow_pct": upper_pct, "lower_shadow_pct": lower_pct,
        "breakout_threshold": breakout_threshold, "breakout": breakout,
        "rsi14": rsi_val if rsi_val is not None else "",
        "score": round(score, 2), "bias": bias,
        "reasons": " | ".join(reasons)
    }

# -------- IO --------
def read_input(path: str, fmt: str) -> List[Dict]:
    fmt = fmt.lower()
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
            else:
                raise ValueError("JSON must be a list or object.")
    elif fmt == "ndjson":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    elif fmt == "csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported --format. Use json | ndjson | csv.")

def write_output(rows: List[Dict], out_path: str, xlsx: bool):
    df = pd.DataFrame(rows)
    if not xlsx:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheet_name = "Scores"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook  = writer.book
        sheet = writer.sheets[sheet_name]

        # Locate columns
        cols = {name: idx for idx, name in enumerate(df.columns)}
        n = len(df)

        # KPI box
        try:
            kpi_text = (f"Score: {df['score'].iloc[-1]}    Bias: {df['bias'].iloc[-1]}\n"
                        f"ΔVWAP%: {df['delta_vwap_pct'].iloc[-1]:.2f}%    Vol×: {df['vol_factor'].iloc[-1]:.2f}")
            fmt = workbook.add_format({"bold": True, "font_size": 12, "align": "left", "valign": "vcenter"})
            sheet.merge_range("N2:R5", kpi_text, fmt)
        except Exception:
            pass

        if n >= 2:
            # Score line chart
            chart1 = workbook.add_chart({"type": "line"})
            chart1.add_series({
                "name":       "Strength Score",
                "categories": [sheet_name, 1, cols.get("period", 0), n, cols.get("period", 0)],
                "values":     [sheet_name, 1, cols.get("score", 0),  n, cols.get("score", 0)],
            })
            chart1.set_title({"name": "5-min Strength Score Trend"})
            chart1.set_y_axis({"name": "Score (0–10)", "min": 0, "max": 10})
            chart1.set_x_axis({"name": "Period"})
            sheet.insert_chart("N7", chart1)
        else:
            # Single-point visualization: scatter with marker & label
            chart1 = workbook.add_chart({"type": "scatter"})
            chart1.add_series({
                "name": "Score",
                "categories": [sheet_name, 1, cols.get("period", 0), 1, cols.get("period", 0)],
                "values":     [sheet_name, 1, cols.get("score", 0),  1, cols.get("score", 0)],
                "marker": {"type": "circle", "size": 8},
                "data_labels": {"value": True},
            })
            chart1.set_title({"name": "Score Snapshot"})
            chart1.set_y_axis({"name": "Score (0–10)", "min": 0, "max": 10})
            chart1.set_x_axis({"name": "Period"})
            sheet.insert_chart("N7", chart1)

        # Vol factor bar/scatter (works for both 1 row and many rows)
        chart2 = workbook.add_chart({"type": "column" if n >= 2 else "scatter"})
        chart2.add_series({
            "name":       "Vol×",
            "categories": [sheet_name, 1, cols.get("period", 0), max(n,1), cols.get("period", 0)],
            "values":     [sheet_name, 1, cols.get("vol_factor", 0),  max(n,1), cols.get("vol_factor", 0)],
            "data_labels": {"value": n == 1}
        })
        chart2.set_title({"name": "Volume Factor"})
        chart2.set_y_axis({"name": "Multiple"})
        sheet.insert_chart("N22", chart2)

        # Conditional formatting for score
        score_col = cols.get("score", None)
        if score_col is not None:
            first_row = 1; last_row = n
            sheet.conditional_format(first_row, score_col, last_row, score_col, {
                "type": "3_color_scale",
                "min_color": "#FF0000", "mid_color": "#FFFF00", "max_color": "#00FF00"
            })

        # Data bars for vol_factor
        vf_col = cols.get("vol_factor", None)
        if vf_col is not None:
            sheet.conditional_format(1, vf_col, n, vf_col, {
                "type": "data_bar",
                "bar_color": "#5DADE2"
            })

        # Freeze header
        sheet.freeze_panes(1, 1)

def main():
    parser = argparse.ArgumentParser(description="Enhanced 5-minute strength scoring CLI (v2, single-row viz)")
    parser.add_argument("--input", "-i", required=True, nargs="+", help="One or more input files (JSON array / NDJSON / CSV)")
    parser.add_argument("--format", "-f", default="json", choices=["json","ndjson","csv"], help="Input file format")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Breakout threshold price (e.g., 144.5)")
    parser.add_argument("--per5avg", type=float, default=None, help="Override per-5min average volume (if given, ignore avg_vol_5d)")
    parser.add_argument("--output", "-o", default=None, help="Output path (.csv or .xlsx). Default: same folder with _scored")
    parser.add_argument("--xlsx", action="store_true", help="Write as .xlsx instead of .csv (adds charts & formatting)")
    parser.add_argument("--trend-window", type=int, default=3, help="Trend consistency lookback bars (default 3)")
    parser.add_argument("--sort-by-period", action="store_true", help="Try to sort combined rows by ET period (if parseable)")
    args = parser.parse_args()


try:
    bars_all = []
    for path in args.input:
        rows = read_input(path, args.format)
        # Attach source file for traceability
        for r in rows:
            r = dict(r)
            r["__source_file__"] = os.path.basename(path)
            bars_all.append(r)
    bars = bars_all
except Exception as e:

        print(f"[ERROR] Failed to read input: {e}", file=sys.stderr)
        sys.exit(2)


# Optional sort by 'period' if present and parseable as timestamp. Otherwise, keep file order.
if args.sort_by_period:
    def _parse_period(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT
    for r in bars:
        r["__period_ts__"] = _parse_period(r.get("period_start_et") or r.get("timestamp") or r.get("time") or r.get("period"))
    df_tmp = pd.DataFrame(bars)
    df_tmp = df_tmp.sort_values(by="__period_ts__", kind="stable")
    bars = df_tmp.drop(columns=["__period_ts__"], errors="ignore").to_dict(orient="records")

results = []
for idx, bar in enumerate(bars):

        prior = bars[max(0, idx-args.trend_window):idx] if idx > 0 else None
        r = strength_score_5min(
            bar,
            prior_bars=prior,
            per5min_avg_vol_override=args.per5avg,
            breakout_threshold=args.threshold,
            trend_window=args.trend_window
        )
        results.append(r)

    # Output path

if args.output:
    out_path = args.output
else:
    if len(args.input) == 1:
        base, ext = os.path.splitext(args.input[0])
        out_path = f"{base}_scored.xlsx" if args.xlsx else f"{base}_scored.csv"
    else:
        out_path = "combined_scored.xlsx" if args.xlsx else "combined_scored.csv"

    try:
        write_output(results, out_path, xlsx=args.xlsx)
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Saved results -> {out_path}")
    # Quick preview
    try:
        preview = pd.DataFrame(results).head(5)
        print(preview.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
