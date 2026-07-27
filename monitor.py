#!/usr/bin/env python3
"""monitor.py — Torn 股市盯盘提醒(只读,不自动下单)。

策略与 analysis/03_backtest.py 回测验证过的参数同源:
  - 抄底(dip):现价 ≤ 过去 W 天最高价 × (1-x) → 提醒买入
  - 止盈/超时:提醒后假定以当时价格入场,涨 y% 提醒卖出,T 天后超时提醒
  - z-score(可选):现价对 N 日均线偏离 < -k 个标准差 → 提醒买入,回归 ≥0 提醒卖出

历史窗口启动时用 data/merged/*.parquet 补齐,之后每分钟轮询 tornsy API。

用法:
  .venv/bin/python monitor.py                     # 默认参数盯全部 35 股
  .venv/bin/python monitor.py --stocks SYM,FHG    # 只盯指定股票
  .venv/bin/python monitor.py --strategies dip    # 只用抄底策略
  .venv/bin/python monitor.py reset SYM           # 重置某股状态(如手动已卖出)

状态文件: analysis/output/monitor_state.json(可手动编辑)
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API_BASE = "https://tornsy.com/api"
DATA_DIR = Path(__file__).resolve().parent / "data" / "merged"
STATE_FILE = Path(__file__).resolve().parent / "analysis" / "output" / "monitor_state.json"

# ── 策略参数(与 03_backtest.py 的样本外优胜参数一致)──
DIP = {"W": 30, "x": 0.01, "y": 0.01, "T_days": 90}
ZSCORE = {"N": 30, "k": 1.0, "T_days": 90}

POLL_INTERVAL = 60          # 秒
REQUEST_GAP = 0.25          # 每次请求间隔(礼貌限速)
WINDOW_DAYS = max(DIP["W"], ZSCORE["N"]) + 5


def fetch_latest(symbol: str) -> tuple[int, float, int] | None:
    """取最新一分钟的 (timestamp, price, total_shares)。"""
    now = int(time.time())
    url = (f"{API_BASE}/{symbol.lower()}?interval=m1"
           f"&from={now - 300}&to={now}&limit=10")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = json.loads(resp.read())
        data = body.get("data") or []
        if not data:
            return None
        ts, price, shares = data[-1]
        return int(ts), float(price), int(shares)
    except Exception as e:
        print(f"  [{symbol}] 请求失败: {e}")
        return None


def seed_history(symbol: str) -> pd.Series:
    """从本地 parquet 读取最近 WINDOW_DAYS 天的日线收盘。"""
    df = pd.read_parquet(DATA_DIR / f"{symbol}.parquet", columns=["timestamp", "price"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    daily = df.set_index("datetime")["price"].resample("1D").last().dropna()
    return daily.tail(WINDOW_DAYS)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def alert(msg: str) -> None:
    print(f"\a[{now_str()}] *** {msg} ***", flush=True)


def run(stocks: list[str], strategies: list[str]) -> None:
    hist = {s: seed_history(s) for s in stocks}
    state = load_state()
    for s in stocks:
        state.setdefault(s, {"position": None})   # position: {strategy, entry_px, entry_ts}

    print(f"[{now_str()}] 盯盘启动: {len(stocks)} 只标的, 策略={strategies}, "
          f"每 {POLL_INTERVAL}s 轮询")
    day_cache_date = None

    while True:
        for s in stocks:
            latest = fetch_latest(s)
            time.sleep(REQUEST_GAP)
            if latest is None:
                continue
            ts, price, _ = latest
            day = datetime.fromtimestamp(ts, tz=timezone.utc).date()

            # 日线窗口每日滚动一次
            if day_cache_date != day:
                for sym in stocks:
                    h = hist[sym]
                    if h.index[-1].date() < day:
                        hist[sym] = h.tail(WINDOW_DAYS)
                day_cache_date = day

            daily = hist[s]
            pos = state[s]["position"]
            dip_win = daily.tail(DIP["W"])
            roll_max = dip_win.max()

            if pos is None:
                if "dip" in strategies and price <= roll_max * (1 - DIP["x"]):
                    alert(f"{s} 抄底信号: ${price:.2f} ≤ {DIP['W']}日高点 "
                          f"${roll_max:.2f} 的 {1-DIP['x']:.0%},建议买入")
                    state[s]["position"] = {"strategy": "dip", "entry_px": price,
                                            "entry_ts": ts}
                    save_state(state)
                elif "zscore" in strategies:
                    zwin = daily.tail(ZSCORE["N"])
                    ma, sd = zwin.mean(), zwin.std()
                    if sd > 0 and (price - ma) / sd < -ZSCORE["k"]:
                        alert(f"{s} z-score 信号: ${price:.2f}, "
                              f"z={(price-ma)/sd:.2f} < -{ZSCORE['k']},建议买入")
                        state[s]["position"] = {"strategy": "zscore",
                                                "entry_px": price, "entry_ts": ts}
                        save_state(state)
            else:
                entry_px = pos["entry_px"]
                held_days = (ts - pos["entry_ts"]) / 86400
                strat = pos["strategy"]
                if strat == "dip":
                    take_profit = price >= entry_px * (1 + DIP["y"])
                else:
                    zwin = daily.tail(ZSCORE["N"])
                    take_profit = (price - zwin.mean()) / max(zwin.std(), 1e-9) >= 0
                if take_profit:
                    alert(f"{s} 止盈信号: ${price:.2f},入场 ${entry_px:.2f},"
                          f"毛收益 {price/entry_px-1:+.2%},建议卖出")
                    state[s]["position"] = None
                    save_state(state)
                elif held_days >= (DIP["T_days"] if strat == "dip" else ZSCORE["T_days"]):
                    alert(f"{s} 超时退出: ${price:.2f},入场 ${entry_px:.2f} "
                          f"({held_days:.0f}天前),毛收益 {price/entry_px-1:+.2%}")
                    state[s]["position"] = None
                    save_state(state)

        time.sleep(POLL_INTERVAL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Torn 股市盯盘提醒")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "reset"])
    ap.add_argument("reset_stock", nargs="?", default=None)
    ap.add_argument("--stocks", type=str, default="",
                    help="逗号分隔,默认全部(不含 TCSE)")
    ap.add_argument("--strategies", type=str, default="dip,zscore",
                    help="dip / zscore / dip,zscore")
    args = ap.parse_args()

    if args.cmd == "reset":
        state = load_state()
        sym = (args.reset_stock or "").upper()
        if sym in state:
            state[sym]["position"] = None
            save_state(state)
            print(f"已重置 {sym} 状态")
        else:
            print(f"状态文件中没有 {sym}")
        return

    if args.stocks:
        stocks = [s.strip().upper() for s in args.stocks.split(",") if s.strip()]
    else:
        stocks = sorted(p.stem for p in DATA_DIR.glob("*.parquet")
                        if p.stem != "TCSE")
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    run(stocks, strategies)


if __name__ == "__main__":
    main()
