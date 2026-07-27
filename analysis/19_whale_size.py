"""19_whale_size.py — 巨鲸交易的美元体量识别、分档与事后收益测算。

先验:巨鲸资金体量 $300B–$8T。total_shares = 全体玩家持仓总量,分钟差分 = 全市场净资金流。

识别局限(重要,贯穿全部结论):
  1) 同一分钟内多人交易互相抵消,只能看到净额——"单笔"实际是"该分钟净打印";
  2) 无身份链接,无法把多分钟的打印归因到同一只巨鲸;
  3) 卖单不可归因(卖出者身份与买入者无法匹配)。
  因此本脚本测的是"巨鲸级交易群体"的事后收益,不是任何个体的真实收益曲线。

无未来函数:打印在 t 分钟收盘后可见,收益从 t+1 分钟价格起算;卖出税 0.1%。

输出 analysis/output/19_whale_size.csv(分档事件研究)/ 19_flow_stats.csv(流量分布+尖峰归因)
     / 19_whale_size.png
用法: cd analysis && ../.venv/bin/python 19_whale_size.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUT_DIR, SELL_TAX, list_stocks, load_stock

KEEP = 1.0 - SELL_TAX
HORIZONS = {"1d": 86400, "3d": 3 * 86400, "7d": 7 * 86400, "14d": 14 * 86400}
TIERS = [(0, 1e9, "<$1B"), (1e9, 1e10, "$1-10B"), (1e10, 1e11, "$10-100B"),
         (1e11, 1e12, "$100B-1T"), (1e12, np.inf, ">$1T")]
FLOW_THRESHOLDS = [1e9, 1e10, 5e10, 1e11, 3e11, 1e12]


def load_minute(sym):
    df = load_stock(sym)
    # pandas 3 索引为 datetime64[s]:view 直接给秒,用分辨率无关写法
    ts = df.index.to_numpy(dtype="datetime64[s]").astype("int64")
    return ts, df["price"].to_numpy(), df["total_shares"].to_numpy()


def fwd_from(ts, px, j, sec):
    """从第 j 分钟起 sec 秒后的收益。"""
    k = np.searchsorted(ts, ts[j] + sec)
    return px[min(k, len(px) - 1)] / px[j] - 1 if k < len(ts) else np.nan


def main() -> None:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    ts_map, px_map, flow_map = {}, {}, {}
    close, shares_d = {}, {}
    price_med = {}

    for sym in stocks:
        ts, px, sh = load_minute(sym)
        ts_map[sym], px_map[sym] = ts, px
        flow_map[sym] = np.diff(sh).astype(np.float64) * px[1:]   # 分钟净美元流(对齐到分钟 1..n-1)
        price_med[sym] = float(np.median(px))
        idx = pd.to_datetime(ts, unit="s", utc=True)
        s = pd.Series(px, index=idx)
        close[sym] = s.resample("1D").last()
        shares_d[sym] = pd.Series(sh, index=idx).resample("1D").last()

    close_df = pd.DataFrame(close).ffill()
    z20 = ((close_df - close_df.rolling(20).mean()) / close_df.rolling(20).std()).shift(1)
    dates = close_df.index
    day_start = np.array([d.timestamp() for d in dates])
    n_days = len(dates)

    # ================= 1. 美元流分布 =================
    print("=== 1. 资金流美元化与分布 ===")
    print(f"各股价格中位数范围: ${min(price_med.values()):,.0f} – ${max(price_med.values()):,.0f} "
          f"(全市场中位 ${np.median(list(price_med.values())):,.0f})")
    all_buy, all_sell = [], []
    thr_rows = []
    for sym in stocks:
        f = flow_map[sym]
        buy, sell = f[f > 0], -f[f < 0]
        all_buy.append(buy)
        all_sell.append(sell)
        row = {"stock": sym}
        for t in FLOW_THRESHOLDS:
            row[f"buy≥${t/1e9:.0f}B"] = int((buy >= t).sum())
            row[f"sell≥${t/1e9:.0f}B"] = int((sell >= t).sum())
        thr_rows.append(row)
    all_buy = np.concatenate(all_buy)
    all_sell = np.concatenate(all_sell)
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUTPUT_DIR / "19_flow_stats.csv", index=False)
    for name, arr in (("买入流", all_buy), ("卖出流", all_sell)):
        print(f"{name}: n={len(arr):,} 中位 ${np.median(arr)/1e6:,.1f}M "
          f"p99 ${np.percentile(arr,99)/1e9:,.1f}B p99.99 ${np.percentile(arr,99.99)/1e9:,.1f}B "
          f"max ${arr.max()/1e9:,.1f}B")
    print("分钟净流阈值笔数(35 股合计):")
    for t in FLOW_THRESHOLDS:
        nb = int((all_buy >= t).sum())
        nsl = int((all_sell >= t).sum())
        ns_stocks = int((thr_df[f"buy≥${t/1e9:.0f}B"] > 0).sum())
        print(f"  ≥${t/1e9:,.0f}B: 买 {nb:,} 笔 / 卖 {nsl:,} 笔,涉及 {ns_stocks} 只股票")

    # ================= 2. 尖峰归因 =================
    print("\n=== 2. 尖峰归因(158 个日增幅>50% 事件)===")
    sh_df = pd.DataFrame(shares_d).ffill()
    dpct = sh_df.pct_change()
    spike_rows = []
    for sym in stocks:
        ev_days = dpct.index[dpct[sym] > 0.5]
        ts, f = ts_map[sym], flow_map[sym]
        for d in ev_days:
            i = sh_df.index.get_loc(d)
            if i < 1 or i + 7 >= len(sh_df):
                continue
            d0, d1 = day_start[i], day_start[i] + 86400
            w = f[np.searchsorted(ts, d0) - 1:np.searchsorted(ts, d1) - 1]
            w = w[np.isfinite(w)]
            buys = np.sort(w[w > 0])[::-1]
            day_net = w.sum()
            top3 = buys[:3].sum() if len(buys) else 0.0
            spike_rows.append({
                "stock": sym, "day": str(d.date()), "day_net_$B": day_net / 1e9,
                "top1_$B": (buys[0] / 1e9 if len(buys) else 0),
                "top3_share_of_net": (top3 / day_net if day_net > 0 else np.nan),
                "n_buy_prints_$1B+": int((buys >= 1e9).sum()),
                "revert_7d": float(sh_df[sym].iloc[i + 7] / sh_df[sym].iloc[i - 1]),
            })
    sp = pd.DataFrame(spike_rows)
    sp.to_csv(OUTPUT_DIR / "19_spike_attribution.csv", index=False)
    few = (sp["top3_share_of_net"] > 0.5).mean()
    print(f"  {len(sp)} 个尖峰:top-3 分钟打印占当日净流入 >50% 的占比 {few:.0%}")
    print(f"  尖峰日最大单笔买入流: 中位 ${sp['top1_$B'].median():,.1f}B,"
          f"最大 ${sp['top1_$B'].max():,.1f}B")
    print(f"  对照先验 $300B–$8T:最大单笔 ${sp['top1_$B'].max():,.0f}B = "
          f"300B 玩家的 {sp['top1_$B'].max()/300:.0%} / 8T 玩家的 {sp['top1_$B'].max()/8000:.1%}")
    print(f"  全样本最大分钟净买入流 ${all_buy.max()/1e9:,.0f}B")

    # ================= 3. 体量分档事件研究 =================
    print("\n=== 3. 体量分档事件研究(单笔分钟净打印) ===")
    # 无条件均值(每 6 小时抽样)
    uncond = {H: [] for H in HORIZONS}
    for sym in stocks:
        ts = ts_map[sym]
        for h in range(3600, len(ts) - 15 * 1440, 360):
            for H, sec in HORIZONS.items():
                v = fwd_from(ts, px_map[sym], h, sec)
                if np.isfinite(v):
                    uncond[H].append(v)
    uncond_mean = {H: float(np.mean(v)) for H, v in uncond.items()}

    z_mat = z20.to_numpy()
    sym_i = {s: i for i, s in enumerate(stocks)}
    tier_rows = []
    for side in ("buy", "sell"):
        for lo, hi, label in TIERS:
            recs = {H: [] for H in HORIZONS}
            zrecs = {"low": {H: [] for H in HORIZONS}, "high": {H: [] for H in HORIZONS}}
            for sym in stocks:
                ts, px, f = ts_map[sym], px_map[sym], flow_map[sym]
                mag = f if side == "buy" else -f
                idx = np.nonzero((mag >= lo) & (mag < hi))[0] + 1   # 打印分钟 → 信号分钟
                idx = idx[idx + 1 < len(ts)]
                if len(idx) == 0:
                    continue
                if label == "<$1B" and len(idx) > 30000:            # 小档抽样控制计算量
                    idx = np.random.default_rng(0).choice(idx, 30000, replace=False)
                day_idx = np.clip(np.searchsorted(day_start, ts[idx]) - 1, 0, n_days - 1)
                zv = z_mat[day_idx, sym_i[sym]]                     # 前一日 z
                for H, sec in HORIZONS.items():
                    k = np.searchsorted(ts, ts[idx] + sec)
                    ok = k < len(ts)
                    r = px[np.minimum(k, len(px) - 1)] / px[idx] - 1
                    recs[H].append(r[ok])
                    zrecs["low"][H].append(r[ok & (zv < -1)])
                    zrecs["high"][H].append(r[ok & (zv >= 0)])
            def stat(v, H):
                v = np.concatenate(v) if v else np.array([])
                if len(v) < 10:
                    return np.nan, np.nan, len(v)
                t = (v.mean() - uncond_mean[H]) / (v.std(ddof=1) / np.sqrt(len(v)))
                return v.mean() * 1e4, t, len(v)
            for grp, store in (("all", recs), ("z<-1", zrecs["low"]), ("z>=0", zrecs["high"])):
                row = {"side": side, "tier": label, "zgrp": grp}
                line = f"  {side}/{label}/{grp}".ljust(24)
                for H in HORIZONS:
                    m, t, n_ = stat(store[H], H)
                    row[f"{H}_bp"], row[f"{H}_t"], row["n"] = m, t, n_
                    line += f"{m:+7.0f}(t{t:+5.1f})".rjust(14)
                tier_rows.append(row)
                print(line)
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(OUTPUT_DIR / "19_whale_size.csv", index=False)

    # ================= 4. 巨鲸单笔收益测算($10B+ 买入,24h 去重) =================
    print("\n=== 4. $10B+ 买入打印的单笔收益(同股 24h 去重) ===")
    p4 = []
    for sym in stocks:
        ts, px, f = ts_map[sym], px_map[sym], flow_map[sym]
        idx = np.nonzero(f >= 1e10)[0] + 1
        idx = idx[idx + 1 < len(ts)]
        if len(idx) == 0:
            continue
        keep, last = [], -10**18
        for j in idx[np.argsort(f[idx - 1])[::-1]]:      # 大的优先,24h 去重
            if ts[j] - last >= 86400:
                keep.append(j)
                last = ts[j]
        z_arr = z20[sym].to_numpy()
        for j in sorted(keep):
            di = np.searchsorted(day_start, ts[j]) - 1
            e_px = px[j]
            row = {"stock": sym, "ts": ts[j], "size_$B": f[j - 1] / 1e9}
            for H, sec in HORIZONS.items():
                v = fwd_from(ts, px, j, sec)
                row[f"net_{H}"] = (1 + v) * KEEP - 1 if np.isfinite(v) else np.nan
            # z 回复退出:z20≥0 或 21 天
            sell_d = None
            for dd in range(di, min(di + 21, n_days)):
                if np.isfinite(z_arr[dd]) and z_arr[dd] >= 0:
                    sell_d = dd
                    break
            if sell_d is None:
                sell_d = min(di + 21, n_days - 1)
            k = np.searchsorted(ts, day_start[sell_d] + 86400) - 1
            if k > j:
                gross = px[min(k, len(px) - 1)] / e_px - 1
                row["net_zexit"] = (1 + gross) * KEEP - 1
                row["hold_zexit_d"] = sell_d - di
            p4.append(row)
    p4 = pd.DataFrame(p4)
    p4.to_csv(OUTPUT_DIR / "19_whale_trades.csv", index=False)
    years = (dates[-1] - dates[0]).days / 365.25
    ev_per_year = len(p4) / years
    print(f"  $10B+ 去重事件: {len(p4)} 个({ev_per_year:.0f}/年),"
          f"规模中位 ${p4['size_$B'].median():,.0f}B,最大 ${p4['size_$B'].max():,.0f}B")
    for col, hold in (("net_7d", 7), ("net_14d", 14), ("net_zexit", p4["hold_zexit_d"].mean())):
        v = p4[col].dropna()
        trades_yr = min(ev_per_year, 365 / hold)
        ann = (1 + v.mean()) ** trades_yr - 1
        print(f"  {col}: 中位 {v.median()*1e4:+.0f}bp 均值 {v.mean()*1e4:+.0f}bp "
              f"胜率 {(v>0).mean():.0%} 均持仓 {hold:.0f}d → 理论年化 {ann*100:.0f}%"
              f"(频率 {ev_per_year:.0f}/年 vs 容量 {365/hold:.0f} 槽/年)")

    # ================= 图 =================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    buy = tier_df[(tier_df.side == "buy")]
    x = np.arange(len(TIERS))
    for gi, (grp, mk) in enumerate((("z<-1", 1), ("z>=0", 1))):
        sub = buy[buy.zgrp == grp].set_index("tier")
        vals = [sub.loc[l, "14d_bp"] for _, _, l in TIERS]
        ax.bar(x + (gi - 0.5) * 0.35, vals, 0.35, label=f"buy {grp}")
    ax.set_xticks(x, [l for _, _, l in TIERS])
    ax.set_ylabel("mean fwd 14d return (bp)")
    ax.set_title("Smart-money test: buy prints by USD size")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    sell = tier_df[(tier_df.side == "sell") & (tier_df.zgrp == "all")].set_index("tier")
    buyall = buy[buy.zgrp == "all"].set_index("tier")
    ax.bar(x - 0.175, [buyall.loc[l, "14d_bp"] for _, _, l in TIERS], 0.35, label="buy (all)")
    ax.bar(x + 0.175, [sell.loc[l, "14d_bp"] for _, _, l in TIERS], 0.35, label="sell (all)")
    ax.set_xticks(x, [l for _, _, l in TIERS])
    ax.set_ylabel("mean fwd 14d return (bp)")
    ax.set_title("Buy vs sell prints by size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "19_whale_size.png", dpi=150)
    print(f"\n输出: {OUTPUT_DIR / '19_whale_size.csv'} / 19_flow_stats.csv / "
          f"19_spike_attribution.csv / 19_whale_trades.csv / 19_whale_size.png")


if __name__ == "__main__":
    main()
