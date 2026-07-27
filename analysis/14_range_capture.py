"""14_range_capture.py — 现实中能捕获全知者收益的百分之几?

逻辑链:方向不可预测,但"哪只股明天波动大"可能可预测(波动聚集)。

A. 波动可预测性(样本外):
   - 过去 k 天(k∈{5,10,20})平均 range 预测明日 range 的横截面 Spearman 秩相关;
   - HAR 回归 range_t ~ lag1 + 5 日均值 + 20 日均值,前 70% 训练 / 后 30% 测试,报 OOS R²;
   - HistGradientBoosting:过去 20 个日 range + z-score + 股本变化,同样划分报 OOS R²。

B. 现实日内捕获策略(严格无未来函数,每日开盘只用截至昨天的数据):
   - 按过去 20 日平均 range 选 top-1 / top-3;
   - 限价买单 = 开盘价 × (1 - c×m),m = 过去 20 日 (开盘-最低)/开盘 的中位数,c∈{0.5,0.75,1.0};
   - 成交后止盈 = 成交价 × (1+t),t∈{0.3%,0.5%,0.8%};当日未触及则收盘市价卖出;
     买单未成交则空仓。全程扣 0.1% 卖出税,逐日复利。

C. 汇总:全知者日均 bp → 秩相关 → 现实日均 bp → 捕获比例。

输出 analysis/output/14_range_capture.csv / 14_vol_predict.csv / 14_range_capture.png
用法: cd analysis && ../.venv/bin/python 14_range_capture.py  (需先跑 13)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from common import OUTPUT_DIR, SELL_TAX, list_stocks, load_stock

K_LIST = (5, 10, 20)
WARMUP = 20
N_LIST = (1, 3)
C_LIST = (0.5, 0.75, 1.0)
T_LIST = (0.003, 0.005, 0.008)


def per_stock_daily(sym: str):
    """返回 (每日统计 DataFrame, {日期: 当日分钟价格 float32 数组})。"""
    df = load_stock(sym)
    day = df.index.floor("D")
    g = df.groupby(day)["price"]
    stats = pd.DataFrame({
        "open": g.first(), "close": g.last(),
        "low": g.min(), "high": g.max(),
        "shares": df.groupby(day)["total_shares"].last(),
    })
    stats["range"] = stats["high"] / stats["low"] - 1
    stats["dip"] = (stats["open"] - stats["low"]) / stats["open"]
    prices = {d: v["price"].to_numpy(dtype=np.float64) for d, v in df.groupby(day)}
    return stats, prices


def xsec_spearman(pred: pd.DataFrame, act: pd.DataFrame) -> pd.Series:
    """逐日横截面 Spearman 秩相关(预测排名 vs 实际排名)。"""
    mask = pred.notna() & act.notna()
    pr = pred.where(mask).rank(axis=1)
    ar = act.where(mask).rank(axis=1)
    pm = pr.sub(pr.mean(axis=1), axis=0)
    am = ar.sub(ar.mean(axis=1), axis=0)
    rho = (pm * am).mean(axis=1) / (pm.pow(2).mean(axis=1).pow(0.5) * am.pow(2).mean(axis=1).pow(0.5))
    return rho.dropna()


def oos_r2(y: np.ndarray, X: np.ndarray, day_arr: np.ndarray, split_day) -> float:
    tr = day_arr < split_day
    te = ~tr
    model = LinearRegression().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main() -> None:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    stats, minute = {}, {}
    for s in stocks:
        stats[s], minute[s] = per_stock_daily(s)

    range_m = pd.DataFrame({s: v["range"] for s, v in stats.items()})
    dip_m = pd.DataFrame({s: v["dip"] for s, v in stats.items()})
    open_m = pd.DataFrame({s: v["open"] for s, v in stats.items()})
    shares_m = pd.DataFrame({s: v["shares"] for s, v in stats.items()})
    days = range_m.index
    print(f"区间: {days[0].date()} → {days[-1].date()} ({len(days)} 天, {len(stocks)} 股)")

    # ================= A. 波动可预测性 =================
    vol_rows = {}
    rhos = {}
    for k in K_LIST:
        pred = range_m.rolling(k).mean().shift(1)
        rho = xsec_spearman(pred, range_m)
        rhos[k] = rho
        t = rho.mean() / (rho.std() / np.sqrt(len(rho)))
        vol_rows[f"spearman_k{k}"] = {"mean_rho": rho.mean(), "t_stat": t, "oos_R2": np.nan}
        print(f"[A] k={k:2d} 平均秩相关 {rho.mean():+.3f}  (t={t:.1f}, n={len(rho)} 天)")

    # HAR: range_t ~ lag1 + mean5 + mean20 (全部 shift(1),只用昨日及以前)
    split_day = days[int(len(days) * 0.7)]
    lag1 = range_m.shift(1)
    m5 = range_m.rolling(5).mean().shift(1)
    m20 = range_m.rolling(20).mean().shift(1)
    feat = pd.concat({"lag1": lag1, "m5": m5, "m20": m20}, axis=1)
    long = feat.stack(future_stack=True).join(range_m.stack(future_stack=True).rename("y")).dropna()
    day_arr = long.index.get_level_values(0)
    X = long[["lag1", "m5", "m20"]].to_numpy()
    y = long["y"].to_numpy()
    r2_har = oos_r2(y, X, day_arr, split_day)
    vol_rows["HAR_oos_R2"] = {"mean_rho": np.nan, "t_stat": np.nan, "oos_R2": r2_har}
    print(f"[A] HAR 回归 OOS R² = {r2_har:.4f} (测试段 {split_day.date()} 起)")

    # GBM: 过去 20 个日 range + z-score + 股本变化
    gbm_feat = {f"lag{i}": range_m.shift(i) for i in range(1, 21)}
    std20 = range_m.rolling(20).std().shift(1)
    gbm_feat["z"] = (lag1 - m20) / std20
    gbm_feat["shares_chg"] = shares_m.pct_change().shift(1)
    gfeat = pd.concat(gbm_feat, axis=1)
    glong = gfeat.stack(future_stack=True).join(range_m.stack(future_stack=True).rename("y")).dropna()
    gday = glong.index.get_level_values(0)
    tr = gday < split_day
    gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    gbm.fit(glong.loc[tr].drop(columns="y"), glong.loc[tr, "y"])
    gpred = gbm.predict(glong.loc[~tr].drop(columns="y"))
    gy = glong.loc[~tr, "y"].to_numpy()
    r2_gbm = float(1 - ((gy - gpred) ** 2).sum() / ((gy - gy.mean()) ** 2).sum())
    vol_rows["GBM_oos_R2"] = {"mean_rho": np.nan, "t_stat": np.nan, "oos_R2": r2_gbm}
    print(f"[A] GBM OOS R² = {r2_gbm:.4f}")

    vol_df = pd.DataFrame(vol_rows).T
    vol_df.index.name = "metric"
    vol_df.to_csv(OUTPUT_DIR / "14_vol_predict.csv")

    # ================= B. 现实日内捕获策略 =================
    trade_days = days[WARMUP:]
    years = (trade_days[-1] - trade_days[0]).days / 365.25
    results = []
    for n_pick in N_LIST:
        for c in C_LIST:
            for tp in T_LIST:
                rets = np.zeros(len(trade_days))
                fills = wins = slots = 0
                for i, d in enumerate(trade_days):
                    window = days[i:i + WARMUP]  # 严格为 d 之前的 20 天
                    mr = range_m.loc[window].mean()
                    md = dip_m.loc[window].median()
                    avail = open_m.columns[open_m.loc[d].notna() & (d in range_m.index)]
                    top = mr[avail].nlargest(n_pick)
                    day_rets = []
                    for sym in top.index:
                        slots += 1
                        px = minute[sym].get(d)
                        o = open_m.loc[d, sym]
                        if px is None or not np.isfinite(o) or not np.isfinite(md[sym]):
                            day_rets.append(0.0)
                            continue
                        limit = o * (1 - c * md[sym])
                        hit = np.nonzero(px <= limit)[0]
                        if len(hit) == 0:
                            day_rets.append(0.0)  # 未成交,空仓
                            continue
                        fills += 1
                        j = hit[0]
                        after = px[j + 1:]
                        tp_hit = np.nonzero(after >= limit * (1 + tp))[0]
                        sell = limit * (1 + tp) if len(tp_hit) else px[-1]
                        net = sell / limit * (1 - SELL_TAX) - 1
                        wins += net > 0
                        day_rets.append(net)
                    rets[i] = np.mean(day_rets)
                eq = np.cumprod(1 + rets)
                total = eq[-1]
                ann = total ** (1 / years) - 1
                peak = np.maximum.accumulate(eq)
                mdd = float((eq / peak - 1).min())
                results.append({
                    "top_n": n_pick, "c": c, "take_profit": tp,
                    "annualized_pct": ann * 100,
                    "mean_daily_bp": rets.mean() * 1e4,
                    "total_multiple": total,
                    "win_rate": wins / max(fills, 1),
                    "fill_rate": fills / max(slots, 1),
                    "max_drawdown": mdd,
                })
                print(f"[B] top{n_pick} c={c} tp={tp:.1%}: 年化 {ann*100:,.1f}%  "
                      f"日均 {rets.mean()*1e4:.2f}bp  胜率 {wins/max(fills,1):.1%}  "
                      f"成交率 {fills/max(slots,1):.1%}  maxDD {mdd:.1%}")

    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "14_range_capture.csv", index=False)

    # ================= C. 汇总结论 =================
    oracle = pd.read_csv(OUTPUT_DIR / "13_oracle_decompose.csv", index_col=0)
    oracle_bp = float(oracle.loc["O_full", "mean_daily_bp"])
    best = res.loc[res["annualized_pct"].idxmax()]
    capture = best["mean_daily_bp"] / oracle_bp
    print("\n=== C. 汇总结论 ===")
    print(f"全知者 O_full 日均净收益: {oracle_bp:.1f}bp (年化 {float(oracle.loc['O_full','annualized_pct']):,.0f}%)")
    print(f"波动可预测性: 平均秩相关 k=5/10/20 = "
          f"{vol_rows['spearman_k5']['mean_rho']:.3f} / {vol_rows['spearman_k10']['mean_rho']:.3f} / "
          f"{vol_rows['spearman_k20']['mean_rho']:.3f};HAR OOS R²={r2_har:.3f},GBM OOS R²={r2_gbm:.3f}")
    print(f"现实最优: top{int(best['top_n'])} c={best['c']} tp={best['take_profit']:.1%} → "
          f"年化 {best['annualized_pct']:,.1f}%,日均 {best['mean_daily_bp']:.2f}bp")
    print(f"捕获比例(日均 bp 口径): {capture:.2%}")

    # 图:秩相关随时间走势(60 日滚动均值)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in K_LIST:
        rhos[k].rolling(60).mean().plot(ax=ax, label=f"k={k}")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_title("Cross-sectional Spearman: past-k-day mean range rank vs next-day range rank (60d rolling)")
    ax.set_ylabel("rank correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_range_capture.png", dpi=150)
    print(f"\n输出: {OUTPUT_DIR / '14_range_capture.csv'} / 14_vol_predict.csv / 14_range_capture.png")


if __name__ == "__main__":
    main()
