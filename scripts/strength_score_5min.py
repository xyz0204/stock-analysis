
import math
from typing import List, Dict, Optional

def compute_per5min_avg_from_5d(avg_vol_5d: Optional[float]) -> Optional[float]:
    if avg_vol_5d is None:
        return None
    try:
        return avg_vol_5d / 78.0
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

def strength_score_5min(bar: Dict, prior_bars: Optional[List[Dict]] = None, per5min_avg_vol_override: Optional[float] = None, breakout_threshold: Optional[float] = None) -> Dict:
    open_p = float(bar.get("period_open", bar.get("session_open")))
    high = float(bar.get("period_high", bar.get("session_high")))
    low = float(bar.get("period_low", bar.get("session_low")))
    close = float(bar.get("period_close", bar.get("session_close")))
    volume = float(bar.get("period_volume", 0.0))
    vwap = float(bar.get("period_vwap", bar.get("session_vwap", close)))
    if per5min_avg_vol_override is not None:
        per5min_avg = per5min_avg_vol_override
    else:
        per5min_avg = compute_per5min_avg_from_5d(bar.get("avg_vol_5d"))
    if not per5min_avg or per5min_avg <= 0:
        per5min_avg = max(volume, 1.0)
    vol_factor = volume / per5min_avg
    candle = classify_candle(open_p, high, low, close)
    score = 5.0
    reasons = []
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
                prev_close = float(prior_bars[-1].get("period_close", prior_bars[-1].get("session_close", close)))
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
    return {
        "ticker": bar.get("ticker"),
        "timestamp": bar.get("period_start_et", bar.get("timestamp")),
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume, "per5min_avg": per5min_avg, "vol_factor": vol_factor,
        "vwap": vwap, "delta_vwap_pct": delta_vwap_pct,
        "candle_pattern": candle["pattern_hint"], "body_pct": body_pct,
        "upper_shadow_pct": upper_pct, "lower_shadow_pct": lower_pct,
        "breakout_threshold": breakout_threshold, "breakout": breakout,
        "score": round(score, 2), "bias": bias, "reasons": reasons
    }
