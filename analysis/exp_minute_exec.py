"""exp_minute_exec.py — 实验6: 分钟级执行时机优化。

方法:
1. 先用小时级框架跑最优策略(h_W32_k1.0_P3_f)得到交易列表
2. 对每笔交易, 从原始分钟数据中提取执行窗口内的价格路径
3. 计算三种执行方式的收益差异:
   a) 当前: 信号下一小时收盘价成交
   b) 最优(不可实现): 执行窗口内的最低价买入/最高价卖出
   c) 现实: 窗口前N分钟内若价格继续向有利方向移动δ则执行, 否则市价

这个后处理分析避免了全量分钟级回测的内存和计算开销。

用法: .venv/bin/python analysis/exp_minute_exec.py
"""

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_minute")
INIT_CAPITAL = 1.0
HOURS_W = 32 * 24
K = 1.0
P = 3
T = 90 * 24


def load_panel_hourly():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1h")
        shares[sym] = df["total_shares"].resample("1h").last().reindex(closes[sym].index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def simulate_get_trades(px, sig):
    """运行策略, 返回交易列表(每笔含entry/exit的UTC时间戳)。"""
    dates = px.index
    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []

    warmup = HOURS_W + 100
    start_i = warmup

    for i in range(start_i, len(dates)):
        today = px.iloc[i]
        today_z = sig["z"].iloc[i]

        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym,
                    "entry_dt": dates[pos[sym]["entry_i"]],
                    "exit_dt": dates[i],
                    "entry_px": today[sym],  # 会被后续覆盖
                    "exit_px": today[sym],
                    "shares": pos[sym]["shares"],
                    "entry_z": pos[sym].get("entry_z", np.nan),
                })
                del pos[sym]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                alloc = cash / (n_slots - j)
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i, "entry_z": today_z[sym]}
                cash -= alloc
        pend_sell, pend_buy = [], []

        held = list(pos.keys())
        for sym in held:
            p = pos[sym]
            held_hours = i - p["entry_i"]
            z_val = today_z[sym]
            exit_now = not np.isnan(z_val) and z_val >= 0
            if exit_now or held_hours >= T:
                pend_sell.append(sym)

        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = today_z.dropna()
            cand = cand[(cand < -K) & ~cand.index.isin(pos)]
            fl = sig["flow"].iloc[i]
            cand = cand[fl.reindex(cand.index) > 0]
            ranked = cand.sort_values()
            pend_buy = list(ranked.index[:slots])

    return trades


def analyze_minute_execution(trades, px_minute):
    """对每笔交易, 分析分钟级执行能改善多少。

    px_minute: {sym: DataFrame with datetime index and price column}
    """
    results = []
    for t in trades:
        sym = t["stock"]
        entry_dt = t["entry_dt"]
        exit_dt = t["exit_dt"]

        if sym not in px_minute:
            continue

        df = px_minute[sym]

        # 执行窗口: 信号在上一个bar的收盘做出, 当前bar的价格在过去60分钟内形成
        # 当前执行价格 = bar close. 窗口 = bar的60分钟
        entry_window_start = entry_dt - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1)
        entry_slice = df[(df.index > entry_window_start) & (df.index <= entry_dt)]

        exit_window_start = exit_dt - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1)
        exit_slice = df[(df.index > exit_window_start) & (df.index <= exit_dt)]

        if len(entry_slice) == 0 or len(exit_slice) == 0:
            continue

        # a) 当前方法: 窗口内最后一分钟收盘价 (=小时收盘价)
        baseline_entry = entry_slice["price"].iloc[-1]
        baseline_exit = exit_slice["price"].iloc[-1]
        baseline_ret = baseline_exit * (1 - SELL_TAX) / baseline_entry - 1

        # b) 最优(不可实现): 窗口内最低价买入/最高价卖出
        best_entry = entry_slice["price"].min()
        best_exit = exit_slice["price"].max()
        perfect_ret = best_exit * (1 - SELL_TAX) / best_entry - 1

        # c) 现实方法1: 等待前N分钟, 若有利方向移动>δ就执行, 否则市价
        # 入场: 等价格跌到<窗口开盘价*(1-δ), 或等满N分钟后市价
        N_min = 10
        delta = 0.0005  # 5bp

        # 入场执行
        entry_open = entry_slice["price"].iloc[0]
        entry_target = entry_open * (1 - delta)
        entry_executed = False
        realistic_entry = baseline_entry  # default
        for j in range(min(N_min, len(entry_slice))):
            p = entry_slice["price"].iloc[j]
            if p <= entry_target:
                realistic_entry = p
                entry_executed = True
                break
        if not entry_executed:
            realistic_entry = entry_slice["price"].iloc[min(N_min - 1, len(entry_slice) - 1)]

        # 出场执行
        exit_open = exit_slice["price"].iloc[0]
        exit_target = exit_open * (1 + delta)
        exit_executed = False
        realistic_exit = baseline_exit
        for j in range(min(N_min, len(exit_slice))):
            p = exit_slice["price"].iloc[j]
            if p >= exit_target:
                realistic_exit = p
                exit_executed = True
                break
        if not exit_executed:
            realistic_exit = exit_slice["price"].iloc[min(N_min - 1, len(exit_slice) - 1)]

        realistic_ret = realistic_exit * (1 - SELL_TAX) / realistic_entry - 1

        # d) 更简单的现实方法: 入场用窗口前半段最低价, 出场用窗口前半段最高价
        half_n = max(1, len(entry_slice) // 2)
        half_entry = entry_slice["price"].iloc[:half_n].min()
        half_exit = exit_slice["price"].iloc[:half_n].max()
        half_ret = half_exit * (1 - SELL_TAX) / half_entry - 1

        results.append({
            "stock": sym,
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "baseline_ret_bp": baseline_ret * 1e4,
            "perfect_ret_bp": perfect_ret * 1e4,
            "realistic_ret_bp": realistic_ret * 1e4,
            "half_ret_bp": half_ret * 1e4,
            "entry_improve_bp": (baseline_entry - best_entry) / baseline_entry * 1e4,
            "exit_improve_bp": (best_exit - baseline_exit) / baseline_exit * 1e4,
            "window_minutes": len(entry_slice),
        })

    return pd.DataFrame(results)


def load_minute_prices():
    """加载分钟级价格, 只保留价格列以节省内存。"""
    stocks = [s for s in list_stocks() if s != "TCSE"]
    px = {}
    for sym in stocks:
        df = load_stock(sym)
        px[sym] = df[["price"]]
    return px


def main():
    print("Step 1: 运行小时级策略获取交易列表...")
    px_h, sh_h = load_panel_hourly()
    sig = {}
    sig["z"] = (px_h - px_h.rolling(HOURS_W).mean()) / px_h.rolling(HOURS_W).std()
    sig["flow"] = sh_h.pct_change(5 * 24)
    trades = simulate_get_trades(px_h, sig)
    print(f"  获取 {len(trades)} 笔交易")

    print("Step 2: 加载分钟级价格数据...")
    px_min = load_minute_prices()
    print(f"  加载完成 ({len(px_min)} 只股票)")

    print("Step 3: 分析分钟级执行改善...")
    res = analyze_minute_execution(trades, px_min)
    print(f"  分析 {len(res)} 笔有效交易")

    # 统计
    print("\n=== 分钟级执行分析 ===")
    print(f"总交易数: {len(res)}")
    print(f"\n每笔平均收益(bp):")
    print(f"  当前(小时收盘):     {res.baseline_ret_bp.mean():.1f} bp")
    print(f"  最优(最低买最高卖): {res.perfect_ret_bp.mean():.1f} bp  (上限)")
    print(f"  现实(10min限价):    {res.realistic_ret_bp.mean():.1f} bp")
    print(f"  现实(前半段最优):   {res.half_ret_bp.mean():.1f} bp")

    print(f"\n入场改善 (买入价比小时收盘低多少):")
    print(f"  最优:  {res.entry_improve_bp.mean():.1f} bp")
    print(f"  中位:  {res.entry_improve_bp.median():.1f} bp")
    print(f"  P75:   {res.entry_improve_bp.quantile(0.75):.1f} bp")
    print(f"  P90:   {res.entry_improve_bp.quantile(0.90):.1f} bp")

    print(f"\n出场改善 (卖出价比小时收盘高多少):")
    print(f"  最优:  {res.exit_improve_bp.mean():.1f} bp")
    print(f"  中位:  {res.exit_improve_bp.median():.1f} bp")
    print(f"  P75:   {res.exit_improve_bp.quantile(0.75):.1f} bp")
    print(f"  P90:   {res.exit_improve_bp.quantile(0.90):.1f} bp")

    # 年化估算
    n_trades = len(res)
    avg_baseline = res.baseline_ret_bp.mean()
    avg_perfect = res.perfect_ret_bp.mean()
    avg_realistic = res.realistic_ret_bp.mean()
    avg_half = res.half_ret_bp.mean()

    # 假设3年期间(n_trades笔), 计算年化差异
    # 每笔改善bp × 年化笔数 ≈ 年化bp改善
    annual_trades = n_trades / 3  # 平均每年交易数
    improve_perfect_bp_per_year = (avg_perfect - avg_baseline) * annual_trades / 1e4
    improve_realistic_bp_per_year = (avg_realistic - avg_baseline) * annual_trades / 1e4
    improve_half_bp_per_year = (avg_half - avg_baseline) * annual_trades / 1e4

    print(f"\n年化估算增益 (基于年均 {annual_trades:.0f} 笔):")
    print(f"  最优(不可实现): +{improve_perfect_bp_per_year*100:.1f}pp")
    print(f"  现实(10min限价): +{improve_realistic_bp_per_year*100:.1f}pp")
    print(f"  现实(前半段最优): +{improve_half_bp_per_year*100:.1f}pp")

    # 更多现实变体: 不同N分钟和δ
    print("\n=== 参数敏感性: 限价等待时间 N 和触发阈值 δ ===")
    for N in [5, 10, 15, 30]:
        for delta in [0.0003, 0.0005, 0.001]:
            improvements = []
            for _, t in res.iterrows():
                sym = t["stock"]
                df = px_min[sym]
                entry_dt = t["entry_dt"]
                exit_dt = t["exit_dt"]
                entry_win_start = entry_dt - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1)
                exit_win_start = exit_dt - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1)
                entry_slice = df[(df.index > entry_win_start) & (df.index <= entry_dt)]
                exit_slice = df[(df.index > exit_win_start) & (df.index <= exit_dt)]
                if len(entry_slice) == 0 or len(exit_slice) == 0:
                    continue

                entry_open = entry_slice["price"].iloc[0]
                exit_open = exit_slice["price"].iloc[0]
                baseline_e = entry_slice["price"].iloc[-1]
                baseline_x = exit_slice["price"].iloc[-1]

                # 入场
                e_exec = baseline_e
                for j in range(min(N, len(entry_slice))):
                    if entry_slice["price"].iloc[j] <= entry_open * (1 - delta):
                        e_exec = entry_slice["price"].iloc[j]
                        break
                else:
                    e_exec = entry_slice["price"].iloc[min(N - 1, len(entry_slice) - 1)]

                # 出场
                x_exec = baseline_x
                for j in range(min(N, len(exit_slice))):
                    if exit_slice["price"].iloc[j] >= exit_open * (1 + delta):
                        x_exec = exit_slice["price"].iloc[j]
                        break
                else:
                    x_exec = exit_slice["price"].iloc[min(N - 1, len(exit_slice) - 1)]

                ret = x_exec * (1 - SELL_TAX) / e_exec - 1
                base_ret = baseline_x * (1 - SELL_TAX) / baseline_e - 1
                improvements.append((ret - base_ret) * 1e4)

            if improvements:
                avg_imp = np.mean(improvements)
                annual_imp = avg_imp * annual_trades / 1e4 * 100
                print(f"  N={N:2d}min δ={delta*10000:.0f}bp: 每笔+{avg_imp:+.1f}bp → 年化+{annual_imp:+.1f}pp")

    res.to_csv(OUTDIR / "minute_analysis.csv", index=False)
    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
