"""15_direction_sweep.py — 方向准确率门槛 & 限价抄底的过夜变体。

Part A — 方向准确率门槛(蒙特卡洛):
  模拟"准确率 a 的选股方向预测器":每天以概率 a 选中当日涨幅最大的股票做多
  (开盘买收盘卖,扣 0.1% 卖出税),以概率 1-a 从其余 34 只里随机选一只。
  a 从 0.5 → 1.0 步长 0.02,每个 a 跑 200 个种子取年化均值。
  参考线:25%(z 轮动)/ 282%(O_stock_only)/ 1318%(完美方向)。
  插值报告:达到 25% / 100% / 282% 年化各需要 a≈多少。

Part B — 14 号限价抄底策略的过夜变体(验证"时间尺度匹配"假设):
  只用 14 号最优参数(top1, c=1.0, tp=0.8%):开盘挂限价单 开盘价×(1-m),
  m = 过去 20 日 (开盘-最低)/开盘 中位数。成交后不再当日强制平仓,改为持有到
  ①止盈 0.8% 触发,或 ②价格回到 20 日均线(每日用截至昨日的收盘均线,无未来函数),
  或 ③持有满 13 天收盘强平。持仓期间不开新仓。

输出 analysis/output/15_direction_sweep.csv / .png
用法: cd analysis && ../.venv/bin/python 15_direction_sweep.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUTPUT_DIR, SELL_TAX, list_stocks, load_stock

A_GRID = np.round(np.arange(0.0, 1.001, 0.02), 2)   # 门槛插值需要 a<0.5 段,扫全程
N_SEEDS = 200
WARMUP = 20
TOP_N, C, TP = 1, 1.0, 0.008   # 14 号最优参数组
MAX_HOLD = 13                  # 最大持有天数(与慢回复半衰期同量级)

REFS = [(25, "z-rotation (real) 25%"), (282, "O_stock_only 282%"), (1318, "perfect direction 1318%")]


def per_stock_daily(sym: str):
    df = load_stock(sym)
    day = df.index.floor("D")
    g = df.groupby(day)["price"]
    stats = pd.DataFrame({
        "open": g.first(), "close": g.last(), "low": g.min(), "high": g.max(),
    })
    stats["range"] = stats["high"] / stats["low"] - 1
    stats["dip"] = (stats["open"] - stats["low"]) / stats["open"]
    prices = {d: v["price"].to_numpy() for d, v in df.groupby(day)}
    return stats, prices


def sweep_direction(oc: pd.DataFrame, years: float) -> pd.DataFrame:
    """Part A:年化(a) 蒙特卡洛均值曲线。"""
    oc_np = oc.to_numpy()
    n_days, n_stocks = oc_np.shape
    top = np.nanargmax(oc_np, axis=1)          # 每日真实涨幅最大的股票
    rows = []
    for a in A_GRID:
        anns = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            hit = rng.random(n_days) < a
            alt = rng.integers(0, n_stocks - 1, n_days)
            alt = np.where(alt >= top, alt + 1, alt)   # 从"其余股票"里均匀选
            sel = np.where(hit, top, alt)
            ret = oc_np[np.arange(n_days), sel]
            net = (1 + ret) * (1 - SELL_TAX) - 1
            total = np.prod(1 + net)
            anns.append((total ** (1 / years) - 1) * 100)
        rows.append({"a": a, "annualized_pct": np.mean(anns), "std": np.std(anns)})
    return pd.DataFrame(rows)


def overnight_variant(stats, minute, range_m, dip_m, open_m, close_m, days, years):
    """Part B:限价抄底 + 持有到均线回复(最多 13 天)。"""
    ma20 = close_m.rolling(WARMUP).mean().shift(1)   # 每日开盘时已知的 20 日均线
    trade_days = days[WARMUP:]
    rets = np.zeros(len(trade_days))
    pos = None          # (sym, entry, tp_price, entry_i)
    trades = wins = fills = slots = 0
    hold_days = []

    for i, d in enumerate(trade_days):
        gi = i + WARMUP
        if pos is not None:
            sym, entry, tp_price, entry_i = pos
            px = minute[sym].get(d)
            ma = ma20.loc[d, sym]
            held = i - entry_i + 1
            sell = None
            if px is not None:
                # 均线退出只在均线高于成本时有意义(回复向上);
                # 若均线低于成本,跌破均线不是"回复",不据此卖出(z 策略同理:等 z≥0 或超时)
                ma_exit = ma if (np.isfinite(ma) and ma > entry) else np.inf
                cross = np.nonzero(px >= min(tp_price, ma_exit))[0]
                if len(cross):
                    j = cross[0]
                    sell = tp_price if px[j] >= tp_price else px[j]  # 止盈限价单 / 均线市价单
            if sell is None and held >= MAX_HOLD and px is not None:
                sell = px[-1]                                        # 超时收盘强平
            if sell is not None:
                net = sell / entry * (1 - SELL_TAX) - 1
                rets[i] = net
                wins += net > 0
                hold_days.append(held)
                pos = None
            continue

        # 空仓:按过去 20 日平均 range 选 top1,挂限价买单
        window = days[gi - WARMUP:gi]
        mr = range_m.loc[window].mean()
        md = dip_m.loc[window].median()
        avail = open_m.columns[open_m.loc[d].notna()]
        top = mr[avail].nlargest(TOP_N)
        if len(top) == 0:
            continue
        sym = top.index[0]
        slots += 1
        px = minute[sym].get(d)
        o = open_m.loc[d, sym]
        if px is None or not np.isfinite(o) or not np.isfinite(md[sym]):
            continue
        limit = o * (1 - C * md[sym])
        hit = np.nonzero(px <= limit)[0]
        if len(hit) == 0:
            continue                                       # 未成交,次日重新选股
        fills += 1
        j = hit[0]
        tp_price = limit * (1 + TP)
        ma = ma20.loc[d, sym]
        # 成交当日剩余时间仍可能触发止盈/均线退出
        sell = None
        ma_exit = ma if (np.isfinite(ma) and ma > limit) else np.inf
        after = px[j + 1:]
        cross = np.nonzero(after >= min(tp_price, ma_exit))[0]
        if len(cross):
            k = cross[0]
            sell = tp_price if after[k] >= tp_price else after[k]
        if sell is not None:
            net = sell / limit * (1 - SELL_TAX) - 1
            rets[i] = net
            trades += 1
            wins += net > 0
            hold_days.append(1)
        else:
            pos = (sym, limit, tp_price, i)
            trades += 1

    eq = np.cumprod(1 + rets)
    total = eq[-1]
    ann = (total ** (1 / years) - 1) * 100
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return {
        "annualized_pct": ann, "mean_daily_bp": rets.mean() * 1e4,
        "total_multiple": total, "trades": trades, "fill_rate": fills / max(slots, 1),
        "win_rate": wins / max(trades, 1), "avg_hold_days": float(np.mean(hold_days)) if hold_days else np.nan,
        "max_drawdown": mdd, "open_position_at_end": pos is not None,
    }


def main() -> None:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    stats, minute = {}, {}
    for s in stocks:
        stats[s], minute[s] = per_stock_daily(s)
    open_m = pd.DataFrame({s: v["open"] for s, v in stats.items()})
    close_m = pd.DataFrame({s: v["close"] for s, v in stats.items()})
    range_m = pd.DataFrame({s: v["range"] for s, v in stats.items()})
    dip_m = pd.DataFrame({s: v["dip"] for s, v in stats.items()})
    oc = close_m / open_m - 1
    days = oc.index
    years = (days[-1] - days[0]).days / 365.25
    print(f"区间: {days[0].date()} → {days[-1].date()} ({years:.2f} 年, {len(days)} 天, {len(stocks)} 股)")

    # ---------- Part A ----------
    curve = sweep_direction(oc, years)
    curve.to_csv(OUTPUT_DIR / "15_direction_sweep.csv", index=False)
    print("\n=== A. 方向准确率门槛(200 种子均值)===")
    for _, r in curve.iloc[::5].iterrows():
        print(f"  a={r['a']:.2f}: {r['annualized_pct']:,.0f}%")
    print(f"  a=1.00: {curve.iloc[-1]['annualized_pct']:,.0f}%")
    for target, label in REFS[:1] + [(100, "100%"), (282, "282%")]:
        above = curve[curve["annualized_pct"] >= target]
        if len(above) == 0:
            print(f"  达到 {target}%/年: a=1.0 也不够({curve.iloc[-1]['annualized_pct']:,.0f}%)")
        else:
            i = above.index[0]
            if i == 0:
                a_need = curve.loc[i, "a"]
            else:
                x0, y0 = curve.loc[i - 1, ["a", "annualized_pct"]]
                x1, y1 = curve.loc[i, ["a", "annualized_pct"]]
                a_need = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            print(f"  达到 {target}%/年 需要 a ≈ {a_need:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve["a"], curve["annualized_pct"], "o-", ms=4, label="direction predictor (MC mean)")
    for v, label in REFS:
        ax.axhline(v, ls="--", alpha=0.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("accuracy a (prob. of picking the day's top gainer)")
    ax.set_ylabel("annualized return (%, log scale)")
    ax.set_title("How accurate must a direction/stock picker be?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "15_direction_sweep.png", dpi=150)

    # ---------- Part B ----------
    years_b = (days[-1] - days[WARMUP]).days / 365.25
    res = overnight_variant(stats, minute, range_m, dip_m, open_m, close_m, days, years_b)
    print("\n=== B. 限价抄底过夜变体(top1, c=1.0, tp=0.8%,持有到 20 日均线/止盈/13 天)===")
    for k, v in res.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")
    pd.DataFrame([res]).to_csv(OUTPUT_DIR / "15_overnight_variant.csv", index=False)
    print(f"\n输出: {OUTPUT_DIR / '15_direction_sweep.csv'} / .png / 15_overnight_variant.csv")


if __name__ == "__main__":
    main()
