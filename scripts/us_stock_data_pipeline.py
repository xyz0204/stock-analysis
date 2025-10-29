#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Data Pipeline - CSV + Live (yfinance + Finnhub)
--------------------------------------------------------
新增：
  --yesterday_only   只输出昨日一行（仍拉足历史用于 MA/RSI 计算）

历史批处理：
  python us_stock_data_pipeline.py --tickers OKLO LEU --days 90 --intraday_days 5
只输出昨日（推荐 days ≥ 60 保证指标充分）：
  python us_stock_data_pipeline.py --tickers OKLO LEU --days 60 --intraday_days 5 --yesterday_only
实时（需 FINNHUB_API_KEY）：
  python us_stock_data_pipeline.py --tickers OKLO LEU --live 1m
调试：
  python us_stock_data_pipeline.py --tickers AAPL --days 5 --intraday_days 1 --debug

依赖：
  pip install yfinance pandas numpy requests finnhub-python
"""
import argparse
import datetime as dt
import os
import sys
import time
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------- yfinance --------------------
try:
    import yfinance as yf
except Exception:
    print("[ERROR] 请先安装 yfinance：pip install yfinance", file=sys.stderr)
    sys.exit(1)

# -------------------- 工具：列名归一化 --------------------
def _norm_key(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join([str(x) for x in tup if x is not None]).strip() for tup in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def pick_column(cols: List[str], *candidates: str) -> Optional[str]:
    norm_map = {_norm_key(c): c for c in cols}
    for cand in candidates:  # 精确
        k = _norm_key(cand)
        if k in norm_map:
            return norm_map[k]
    for cand in candidates:  # 宽松
        k = _norm_key(cand)
        for c in cols:
            nk = _norm_key(c)
            if nk.startswith(k) or k in nk:
                return c
    return None

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_columns(df)
    cols = list(df.columns)

    open_col  = pick_column(cols, "Open")
    high_col  = pick_column(cols, "High")
    low_col   = pick_column(cols, "Low")
    close_col = pick_column(cols, "Close", "Adj Close", "AdjClose")
    vol_col   = pick_column(cols, "Volume", "Vol", "TotalVolume")
    adj_col   = pick_column(cols, "Adj Close", "AdjClose")

    out = pd.DataFrame(index=df.index)
    if open_col:  out["Open"]   = pd.to_numeric(df[open_col], errors="coerce")
    if high_col:  out["High"]   = pd.to_numeric(df[high_col], errors="coerce")
    if low_col:   out["Low"]    = pd.to_numeric(df[low_col], errors="coerce")
    if vol_col:   out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce")

    if close_col:
        out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    elif adj_col:
        out["Close"] = pd.to_numeric(df[adj_col], errors="coerce")

    if adj_col:
        out["AdjClose"] = pd.to_numeric(df[adj_col], errors="coerce")

    return out.dropna(how="all", axis=1)

# -------------------- 指标 --------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).ffill()

def moving_averages(close: pd.Series, windows=(5, 10, 20, 30, 60)) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    out = pd.DataFrame(index=close.index)
    for w in windows:
        out[f"MA{w}"] = close.rolling(w).mean()
    return out

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (cum_tp_vol / cum_vol).ffill()

def first_n_minutes_volume(intra: pd.DataFrame, n_minutes: int = 30) -> float:
    if intra.empty:
        return np.nan
    if intra.index.tz is not None:
        intra_local = intra.copy()
        intra_local.index = intra_local.index.tz_convert("US/Eastern")
    else:
        intra_local = intra.tz_localize("UTC").tz_convert("US/Eastern")
    last_date = intra_local.index[-1].date()
    start = pd.Timestamp.combine(pd.Timestamp(last_date), pd.Timestamp("09:30").time()).tz_localize("US/Eastern")
    end = start + pd.Timedelta(minutes=n_minutes)
    window = intra_local[(intra_local.index >= start) & (intra_local.index < end)]
    return float(window["Volume"].sum()) if "Volume" in window.columns else np.nan

def prev_day_pivots_safe(daily: pd.DataFrame) -> Tuple[float, float, float]:
    if len(daily) < 2:
        return np.nan, np.nan, np.nan
    prev = daily.iloc[-2]
    H = pd.to_numeric(prev.get("High", np.nan), errors="coerce")
    L = pd.to_numeric(prev.get("Low", np.nan), errors="coerce")
    C = pd.to_numeric(prev.get("Close", np.nan), errors="coerce")
    if any(pd.isna(v) for v in (H, L, C)):
        return np.nan, np.nan, np.nan
    p = (H + L + C) / 3.0
    r1 = 2 * p - L
    s1 = 2 * p - H
    return p, r1, s1

# -------------------- 数据获取（yfinance） --------------------
def fetch_daily_yf(symbol: str, start: str, end: str, debug: bool=False) -> pd.DataFrame:
    raw = yf.download(symbol, start=start, end=end, progress=False, interval="1d",
                      auto_adjust=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance 返回空日线数据（{symbol}）")
    if debug:
        print(f"[DEBUG] {symbol} daily raw cols:", list(normalize_columns(raw).columns))
    df = normalize_ohlcv(raw)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"{symbol} 日线缺少 Close；原始列: {list(normalize_columns(raw).columns)}")
    return df

def fetch_intraday_yf(symbol: str, intraday_days: int = 10, interval: str = "5m", debug: bool=False) -> pd.DataFrame:
    intervals_try = [interval, "15m"] if interval != "15m" else ["15m"]
    last_err = None
    for iv in intervals_try:
        for attempt in range(3):
            try:
                raw = yf.download(symbol, period=f"{intraday_days}d", interval=iv,
                                  progress=False, auto_adjust=False, prepost=False, group_by="column")
                if raw is not None and not raw.empty:
                    if debug:
                        print(f"[DEBUG] {symbol} {iv} raw cols:", list(normalize_columns(raw).columns))
                    df = normalize_ohlcv(raw)
                    if df.empty or "Close" not in df.columns:
                        last_err = RuntimeError(f"{symbol} {iv} 缺少 Close；原始列: {list(normalize_columns(raw).columns)}")
                    else:
                        df.attrs["interval_used"] = iv
                        return df
                else:
                    last_err = RuntimeError(f"{symbol} {iv} 返回空数据")
            except Exception as e:
                last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"无法获取 {symbol} 的分钟数据（已尝试 5m/15m）：{last_err}")

# -------------------- 构建表 --------------------
def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    mas = moving_averages(out["Close"], windows=(5,10,20,30,60))
    out = pd.concat([out, mas], axis=1)
    out["RSI14"] = rsi(out["Close"], 14)
    return out

def build_intraday_metrics(df_intra: pd.DataFrame) -> pd.DataFrame:
    out = df_intra.copy()
    if all(c in out.columns for c in ["High", "Low", "Close", "Volume"]):
        out["VWAP"] = vwap(out)
    else:
        out["VWAP"] = np.nan
    return out

def summarize_signals(symbol: str, daily_for_signal: pd.DataFrame, intra: Optional[pd.DataFrame]) -> pd.DataFrame:
    valid = daily_for_signal.copy()
    valid["Close"] = pd.to_numeric(valid["Close"], errors="coerce")
    valid = valid.dropna(subset=["Close"])
    if valid.empty:
        raise RuntimeError(f"{symbol}: 没有有效的 Close 值可用于生成 Signals")

    last = valid.iloc[-1]

    def as_float(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(x) if pd.notna(x) else np.nan

    ma_vals = {f"MA{w}": as_float(last.get(f"MA{w}", np.nan)) for w in (5,10,20,30,60)}
    rsi14 = as_float(last.get("RSI14", np.nan))
    last_close = as_float(last.get("Close", np.nan))

    p, r1, s1 = prev_day_pivots_safe(valid)

    first30 = np.nan
    if intra is not None and not intra.empty and "Volume" in intra.columns:
        try:
            first30 = first_n_minutes_volume(intra, 30)
        except Exception:
            first30 = np.nan

    avg5_vol = np.nan
    if "Volume" in valid.columns:
        vol5 = pd.to_numeric(valid["Volume"], errors="coerce").tail(5)
        if not vol5.dropna().empty:
            avg5_vol = float(vol5.mean())

    ratio = np.nan
    if pd.notna(first30) and pd.notna(avg5_vol) and avg5_vol != 0:
        ratio = float(first30 / avg5_vol)

    row = {
        "Symbol": symbol,
        "LastClose": last_close,
        "MA5": ma_vals["MA5"], "MA10": ma_vals["MA10"], "MA20": ma_vals["MA20"],
        "MA30": ma_vals["MA30"], "MA60": ma_vals["MA60"],
        "RSI14": rsi14,
        "Pivot_P": p, "Pivot_R1": r1, "Pivot_S1": s1,
        "First30mVol_today": float(first30) if pd.notna(first30) else np.nan,
        "Avg5D_Volume": float(avg5_vol) if pd.notna(avg5_vol) else np.nan,
        "First30mVol_vs_5D": ratio,
    }
    return pd.DataFrame([row])

# -------------------- 主管道 --------------------
def run_pipeline(tickers: List[str], days: int, intraday_days: int, debug: bool=False, yesterday_only: bool=False):
    # 为了保证指标计算，建议 days 取较大值；但按用户指定执行
    start_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    end_date = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    all_signals = []
    logs = []

    for sym in tickers:
        print(f"\n=== {sym} ===")
        # 日线
        try:
            daily_raw = fetch_daily_yf(sym, start_date, end_date, debug=debug)
            daily_full = build_daily_metrics(daily_raw).copy()  # 完整历史（用于生成 Signals）
            daily_full = daily_full.dropna(subset=["Close"])

            # 选择最终保存的日线（是否只留昨日）
            daily_to_save = daily_full
            if yesterday_only and not daily_full.empty:
                y = (dt.date.today() - dt.timedelta(days=1)).isoformat()
                d_y = daily_full[daily_full.index.strftime("%Y-%m-%d") == y]
                daily_to_save = d_y if not d_y.empty else daily_full.tail(1)

            # 列顺序
            cols_order = [c for c in ["Open","High","Low","Close","AdjClose","Volume",
                                      "MA5","MA10","MA20","MA30","MA60","RSI14"] if c in daily_to_save.columns]
            daily_to_save = daily_to_save[cols_order]
            daily_to_save.to_csv(f"{sym}_Daily.csv", index=True)
            tag = "（仅昨日）" if yesterday_only else "（含 MA/RSI）"
            print(f"✔ 已保存 {sym}_Daily.csv {tag}")
        except Exception as e:
            msg = f"{sym} daily 错误: {e}"
            logs.append(msg); print("⚠", msg)
            continue

        # 分钟
        intra = None
        try:
            intra = fetch_intraday_yf(sym, intraday_days, "5m", debug=debug)
            interval_used = intra.attrs.get("interval_used", "5m")
            intra = build_intraday_metrics(intra)
            out_name = f"{sym}_{interval_used}.csv" if interval_used != "5m" else f"{sym}_5m.csv"
            intra.to_csv(out_name, index=True)
            print(f"✔ 已保存 {out_name}")
        except Exception as e:
            msg = f"{sym} intraday 错误: {e}"
            logs.append(msg); print("⚠", msg)

        # Signals（始终用完整历史 daily_full 生成，避免只留昨日时丢失上下文）
        try:
            sig = summarize_signals(sym, daily_full, intra)
            sig.to_csv(f"{sym}_Signals.csv", index=False)
            all_signals.append(sig)
            print(f"✔ 已保存 {sym}_Signals.csv")
        except Exception as e:
            msg = f"{sym} signals 错误: {e}"
            logs.append(msg); print("⚠", msg)

    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        combined.to_csv("All_Signals.csv", index=False)
        print("✔ 已保存 All_Signals.csv")

    if logs:
        with open("LOGS.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        print("⚠ 一些标的出现错误，详情见 LOGS.txt")

# -------------------- Finnhub 实时模式 --------------------
def parse_live_interval(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("s"):
        return int(s[:-1])
    if s.endswith("m"):
        return int(s[:-1]) * 60
    raise ValueError("live 仅支持 '10s' 或 '1m' 格式")

def fetch_live_prices(symbols: List[str], interval_sec: int):
    try:
        import finnhub
    except Exception:
        print("[ERROR] 实时模式需要 finnhub-python：pip install finnhub-python", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 FINNHUB_API_KEY 环境变量")

    client = finnhub.Client(api_key=api_key)
    fname = "Live_" + "_".join(symbols) + ".csv"
    if not os.path.exists(fname):
        pd.DataFrame(columns=["Timestamp", "Symbol", "Price", "Change", "PrevClose"])\
          .to_csv(fname, index=False)

    print(f"🚀 开始实时监控: {symbols}（间隔 {interval_sec}s）")
    while True:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for sym in symbols:
            try:
                q = client.quote(sym)
                price = q.get("c")
                prev_close = q.get("pc")
                change = (price - prev_close) / prev_close * 100 if (price is not None and prev_close) else np.nan
                print(f"{now} | {sym:<6} {price!s:>8} | Δ {change:+6.2f}%")
                rows.append([now, sym, price, change, prev_close])
            except Exception as e:
                print(f"⚠ {sym} 实时抓取失败: {e}")
        if rows:
            pd.DataFrame(rows, columns=["Timestamp","Symbol","Price","Change","PrevClose"])\
              .to_csv(fname, mode="a", header=False, index=False)
        time.sleep(interval_sec)

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True, help="股票代码列表，如 OKLO LEU AAPL")
    parser.add_argument("--days", type=int, default=90, help="日线天数（默认90）")
    parser.add_argument("--intraday_days", type=int, default=10, help="分钟数据天数（默认10，yfinance 5m 通常≤60天）")
    parser.add_argument("--yesterday_only", action="store_true", help="只输出昨日数据（含完整指标）")
    parser.add_argument("--live", type=str, help="实时监控间隔，如 '10s' 或 '1m'")
    parser.add_argument("--debug", action="store_true", help="打印原始列名/调试信息")
    args = parser.parse_args()

    # 实时模式
    if args.live:
        interval_sec = parse_live_interval(args.live)
        fetch_live_prices(args.tickers, interval_sec)
        sys.exit(0)

    # 历史批处理
    run_pipeline(args.tickers, args.days, args.intraday_days, debug=args.debug, yesterday_only=args.yesterday_only)
