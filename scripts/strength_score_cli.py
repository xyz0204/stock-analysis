
# -*- coding: utf-8 -*-
"""
5-minute strength scoring CLI

Usage examples:
  python strength_score_cli.py --input oklo_5min.json --format json --threshold 144.5 --xlsx
  python strength_score_cli.py -i data.ndjson -f ndjson -t 144.5
  python strength_score_cli.py -i bars.csv -f csv --per5avg 280000 --output scored.xlsx --xlsx
"""

import json
import math
import argparse
from typing import List, Dict, Optional
import pandas as pd
import os

def compute_per5min_avg_from_5d(avg_vol_5d: Optional[float]) -> Optional[float]:
    if avg_vol_5d is None:
        return None
    try:
        return float(avg_vol_5d) / 78.0
    except Exception:
        return None

def classify_candle(open_p: float, high: float, low: float, close: float) -> Dict[str, float]:
    rng = max(high - low, 1e-9)
    body = abs(close - open_p)
    upper = high - max(open_p, close)
    lower = min(open_p, close) - low
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
    breakout_threshold: Optional[float] = None
) -> Dict:
    open_p = float(bar.get("period_open", bar.get("session_open", bar.get("open", 0.0))))
    high = float(bar.get("period_high", bar.get("session_high", bar.get("high", open_p))))
    low = float(bar.get("period_low", bar.get("session_low", bar.get("low", open_p))))
    close = float(bar.get("period_close", bar.get("session_close", bar.get("close", open_p))))
    volume = float(bar.get("period_volume", bar.get("volume", 0.0)))
    vwap = float(bar.get("period_vwap", bar.get("session_vwap", bar.get("vwap", close))))
    avg_vol_5d = bar.get("avg_vol_5d", None)

    if per5min_avg_vol_override is not None:
        per5min_avg = per5min_avg_vol_override
    else:
        per5min_avg = compute_per5min_avg_from_5d(avg_vol_5d)
    if not per5min_avg or per5min_avg <= 0:
        per5min_avg = max(volume, 1.0)

    vol_factor = volume / per5min_avg

    candle = classify_candle(open_p, high, low, close)

    score = 5.0
    reasons: List[str] = []

    if vol_factor >= 3.0:
        score += 2.0; reasons.append(f"放量{vol_factor:.1f}倍（≥3x）")
    elif vol_factor >= 1.5:
        score += 1.0; reasons.append(f"放量{vol_factor:.1f}倍（≥1.5x）")
    elif vol_factor <= 0.4:
        score -= 2.0; reasons.append(f"严重缩量{vol_factor:.2f}x（≤0.4x）")
    elif vol_factor <= 0.7:
        score -= 1.0; reasons.append(f"缩量{vol_factor:.2f}x（≤0.7x）")
    else:
        reasons.append(f"量能接近均值{vol_factor:.2f}x")

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
        score += 1.0; reasons.append("长下影阳线（买盘承接）")

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

    score = max(0.0, min(10.0, score))
    if score >= 7.5: bias = "强势"
    elif score >= 5.5: bias = "偏强"
    elif score >= 4.5: bias = "中性"
    elif score >= 3.5: bias = "偏弱"
    else: bias = "弱势"

    period = bar.get("period_start_et") or bar.get("timestamp") or bar.get("time") or ""

    return {
        "ticker": bar.get("ticker", ""),
        "period": period,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume, "per5min_avg": per5min_avg, "vol_factor": vol_factor,
        "vwap": vwap, "delta_vwap_pct": delta_vwap_pct,
        "candle_pattern": candle["pattern_hint"],
        "body_pct": body_pct, "upper_shadow_pct": upper_pct, "lower_shadow_pct": lower_pct,
        "breakout_threshold": breakout_threshold, "breakout": breakout,
        "score": round(score, 2), "bias": bias,
        "reasons": " | ".join(reasons)
    }

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
    if xlsx:
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

def main():
    parser = argparse.ArgumentParser(description="5-minute strength scoring CLI")
    parser.add_argument("--input", "-i", required=True, help="Path to input file (JSON array / NDJSON / CSV)")
    parser.add_argument("--format", "-f", default="json", choices=["json","ndjson","csv"], help="Input file format")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Breakout threshold price (e.g., 144.5)")
    parser.add_argument("--per5avg", type=float, default=None, help="Override per-5min average volume (if given, ignore avg_vol_5d)")
    parser.add_argument("--output", "-o", default=None, help="Output path (.csv or .xlsx). Default: same folder with _scored.csv")
    parser.add_argument("--xlsx", action="store_true", help="Write as .xlsx instead of .csv")
    args = parser.parse_args()

    bars = read_input(args.input, args.format)

    results = []
    for idx, bar in enumerate(bars):
        prior = [bars[idx-1]] if idx > 0 else None
        r = strength_score_5min(
            bar,
            prior_bars=prior,
            per5min_avg_vol_override=args.per5avg,
            breakout_threshold=args.threshold
        )
        results.append(r)

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_scored.xlsx" if args.xlsx else f"{base}_scored.csv"

    write_output(results, out_path, xlsx=args.xlsx)
    print(f"Saved results -> {out_path}")
    # Quick preview
    try:
        import pandas as pd
        preview = pd.DataFrame(results).head(5)
        print(preview.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
