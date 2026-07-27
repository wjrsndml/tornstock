"""13_oracle_decompose.py — 日线全知者收益的信息拆解(含未来函数,理论标定)。

把 O_full(~3142%/年)分解为三种信息的贡献:
  - O_full        : 知道哪只股 + 日内路径(复刻 12 号脚本,校验实现)
  - O_stock_only  : 只知道哪只股振幅最大,开盘买收盘卖(收跌则空仓)
  - O_timing_only : 只知道日内路径,不知道选哪只(35 只各自的低买高卖取平均)
  - O_direction   : 只知道每只股今天涨/跌,在收涨股里选涨幅最大的,开盘买收盘卖
  - O_half_timing : 选对股票,但入场=开盘与最低点中点,出场=最高点与收盘中点
  - 噪声全知者    : 以概率 p 从 top-5 里随机选股(而不是选第 1),p 扫 0→0.9

输出 analysis/output/13_oracle_decompose.csv / .png
用法: cd analysis && ../.venv/bin/python 13_oracle_decompose.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SELL_TAX, ensure_out, list_stocks, load_stock

MIN_RANGE = 0.001  # 与 12 号脚本一致:机会 ≤0.1% 不交易
TOP_N = 5          # 噪声全知者的备选池
P_GRID = np.round(np.arange(0.0, 1.0, 0.1), 2)
N_SEEDS = 5


def daily_stats(sym: str) -> pd.DataFrame:
    """每股每日:开/收/低/高、当日最大低买高卖收益(drawup)及对应买入价。"""
    df = load_stock(sym)
    day = df.index.floor("D")
    g = df.groupby(day)["price"]
    stats = pd.DataFrame({
        "open": g.first(), "close": g.last(),
        "low": g.min(), "high": g.max(),
    })
    cummin = g.cummin()                       # 日内截至当前的最低价
    du = df["price"] / cummin - 1.0           # 截至当前的最好低买高卖收益
    dug = du.groupby(day)
    stats["drawup"] = dug.max()
    peak_t = dug.idxmax()                     # drawup 最大时刻
    stats["buy_price"] = cummin.loc[peak_t.to_numpy()].to_numpy()
    stats["sell_price"] = df["price"].loc[peak_t.to_numpy()].to_numpy()
    return stats


def perf(net: np.ndarray, years: float) -> dict:
    eq = np.cumprod(1.0 + net)
    total = float(eq[-1])
    return {
        "annualized_pct": (total ** (1 / years) - 1) * 100,
        "mean_daily_bp": float(np.mean(net)) * 1e4,
        "total_multiple": total,
    }


def main() -> None:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    stats = {s: daily_stats(s) for s in stocks}

    # 对齐成 days × stocks 矩阵
    drawup = pd.DataFrame({s: v["drawup"] for s, v in stats.items()})
    rng_mat = pd.DataFrame({s: v["high"] / v["low"] - 1 for s, v in stats.items()})
    oc = pd.DataFrame({s: v["close"] / v["open"] - 1 for s, v in stats.items()})
    open_m = pd.DataFrame({s: v["open"] for s, v in stats.items()})
    close_m = pd.DataFrame({s: v["close"] for s, v in stats.items()})
    low_m = pd.DataFrame({s: v["low"] for s, v in stats.items()})
    high_m = pd.DataFrame({s: v["high"] for s, v in stats.items()})

    days = drawup.index
    years = (days[-1] - days[0]).days / 365.25
    n_days = len(days)
    print(f"区间: {days[0].date()} → {days[-1].date()} ({years:.2f} 年, {n_days} 天, {len(stocks)} 股)")

    rows: dict[str, dict] = {}

    # --- O_full: 选对股 + 完美择时(复刻 12 号) ---
    du = drawup.to_numpy()
    best = np.nanargmax(du, axis=1)
    du_best = du[np.arange(n_days), best]
    net_full = np.where(du_best > MIN_RANGE, (1 + du_best) * (1 - SELL_TAX) - 1, 0.0)
    rows["O_full"] = perf(net_full, years)

    # --- O_stock_only: 只知哪只振幅最大,开盘买收盘卖(收跌空仓) ---
    r_sel = np.nanargmax(rng_mat.to_numpy(), axis=1)
    oc_sel = oc.to_numpy()[np.arange(n_days), r_sel]
    net = np.where(oc_sel > 0, (1 + oc_sel) * (1 - SELL_TAX) - 1, 0.0)
    rows["O_stock_only"] = perf(net, years)

    # --- O_timing_only: 只知日内路径,35 只各自完美择时取平均 ---
    net_each = np.where(du > MIN_RANGE, (1 + du) * (1 - SELL_TAX) - 1, 0.0)
    rows["O_timing_only"] = perf(np.nanmean(net_each, axis=1), years)

    # --- O_direction: 只知涨跌方向,收涨股中选涨幅最大,开盘买收盘卖 ---
    oc_np = oc.to_numpy()
    oc_masked = np.where(oc_np > 0, oc_np, np.nan)
    has_win = ~np.isnan(oc_masked).all(axis=1)
    g_max = np.nanmax(np.where(has_win[:, None], oc_masked, np.nan), axis=1)
    net = np.where(has_win, (1 + g_max) * (1 - SELL_TAX) - 1, 0.0)
    rows["O_direction"] = perf(net, years)

    # --- O_half_timing: 选对股,入场/出场价格打 5 折精度 ---
    o_sel = open_m.to_numpy()[np.arange(n_days), best]
    c_sel = close_m.to_numpy()[np.arange(n_days), best]
    lo_sel = low_m.to_numpy()[np.arange(n_days), best]
    hi_sel = high_m.to_numpy()[np.arange(n_days), best]
    entry = (o_sel + lo_sel) / 2
    exit_ = (hi_sel + c_sel) / 2
    ok = (du_best > MIN_RANGE) & (exit_ > entry)
    net = np.where(ok, exit_ / entry * (1 - SELL_TAX) - 1, 0.0)
    rows["O_half_timing"] = perf(net, years)

    # --- 噪声全知者:以概率 p 从 top-5 随机选股 ---
    # 每天 top-5 的列位置
    part = np.argpartition(-du, TOP_N - 1, axis=1)[:, :TOP_N]           # 无序 top5
    top5_vals = np.take_along_axis(du, part, axis=1)
    order = np.argsort(-top5_vals, axis=1)
    top5 = np.take_along_axis(part, order, axis=1)                      # 有序 top5 列位置

    noise_ann = []
    for p in P_GRID:
        anns = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            miss = rng.random(n_days) < p
            pick = rng.integers(0, TOP_N, n_days)
            sel = np.where(miss, top5[np.arange(n_days), pick], top5[:, 0])
            du_sel = du[np.arange(n_days), sel]
            net = np.where(du_sel > MIN_RANGE, (1 + du_sel) * (1 - SELL_TAX) - 1, 0.0)
            anns.append(perf(net, years)["annualized_pct"])
        noise_ann.append(float(np.mean(anns)))

    # --- 汇总输出 ---
    summary = pd.DataFrame(rows).T
    summary.index.name = "variant"
    outdir = ensure_out(".")
    summary.to_csv(outdir / "13_oracle_decompose.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== 全知者信息拆解 ===")
    print(summary.to_string())
    print("\n=== 噪声敏感性(年化% vs 选错概率)===")
    for p, a in zip(P_GRID, noise_ann):
        print(f"  p={p:.1f}: {a:,.0f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(P_GRID, noise_ann, "o-", label="noisy oracle (pick from top-5)")
    ax.axhline(rows["O_full"]["annualized_pct"], ls="--", c="gray", label="O_full (perfect pick)")
    ax.set_xlabel("probability of missing the #1 stock (uniform pick from top-5)")
    ax.set_ylabel("annualized return (%)")
    ax.set_title("Oracle sensitivity to stock-picking accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "13_oracle_decompose.png", dpi=150)
    print(f"\n输出: {outdir / '13_oracle_decompose.csv'} / .png")


if __name__ == "__main__":
    main()
