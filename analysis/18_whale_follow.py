"""18_whale_follow.py — 巨鲸跟单策略测试(total_shares = 全体玩家持仓总量)。

分钟级净买入流 = total_shares 分钟差分。全程严格无未来函数:
信号在 t 分钟收盘后可见,最早 t+1 分钟成交;卖出税 0.1%;前 70% 时间 train / 后 30% test。

第一步:分钟资金流结构体检(差分分布、滚动分位数巨鲸事件、尖峰是否回落)。
第二步:事件研究(事件后 1h/6h/1d/3d/7d/14d 收益 vs 无条件,t 值;按事件时 z_20 分组)。
第三步:跟单回测(P1 单仓位,退出规则网格 × z<0 过滤 × q 分位数),
        以及"跟单 + z 超跌双触发"组合策略。
第四步:控制 z 后的增量(OLS:远期收益 ~ z + 巨鲸虚拟变量)。

输出 analysis/output/18_whale_follow.csv / 18_events.csv / 18_whale_follow.png
用法: cd analysis && ../.venv/bin/python 18_whale_follow.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUT_DIR, SELL_TAX, cagr, list_stocks, load_stock, max_drawdown

KEEP = 1.0 - SELL_TAX
WARMUP = 32
Q_LIST = (0.99, 0.995, 0.999)
HOURS_60D = 60 * 24
HORIZONS = {"1h": 3600, "6h": 6 * 3600, "1d": 86400, "3d": 3 * 86400,
            "7d": 7 * 86400, "14d": 14 * 86400}


# ---------------------------------------------------------------- 数据
def load_minute(sym):
    df = load_stock(sym)
    # pandas 3 索引是 datetime64[s]:view("int64") 直接给秒,不能再除 1e9
    ts = df.index.to_numpy(dtype="datetime64[s]").astype("int64")
    px = df["price"].to_numpy()
    sh = df["total_shares"].to_numpy()
    return ts, px, sh


def zscore_panel(mats_close: pd.DataFrame, W=20) -> pd.DataFrame:
    return (mats_close - mats_close.rolling(W).mean()) / mats_close.rolling(W).std()


# ---------------------------------------------------------------- 主流程
def main() -> None:
    stocks = [s for s in list_stocks() if s != "TCSE"]

    # ---------- 第一遍:每股小时资金流、巨鲸事件、事件研究 ----------
    print("=== 第一步:分钟资金流结构体检 ===")
    all_min_diff = []
    events = []            # (stock, hour_end_ts, side, q, prev_z, entry_px, {H: fwd_ret})
    spike_check = []
    daily_close = {}
    ts0_global = None

    for sym in stocks:
        ts, px, sh = load_minute(sym)
        if ts0_global is None:
            ts0_global = ts[0]
        s_px = pd.Series(px, index=pd.to_datetime(ts, unit="s", utc=True))
        s_sh = pd.Series(sh, index=s_px.index)
        daily_close[sym] = s_px.resample("1D").last()

        # 分钟差分分布
        dmin = np.diff(sh)
        nz = dmin[dmin != 0]
        all_min_diff.append(nz[np.abs(nz) < np.percentile(np.abs(nz), 99.9)])

        # 小时资金流 + 滚动分位数事件
        h_sh = s_sh.resample("1h").last().dropna()
        dh = h_sh.diff()
        hour_end_ts = h_sh.index.to_numpy(dtype="datetime64[s]").astype("int64")
        dhv = dh.to_numpy()
        # 事件前一日 z(分类/过滤用,可交易信息)
        # 先占位,z panel 算完后统一回填
        for q in Q_LIST:
            hi = dh.rolling(HOURS_60D).quantile(q).shift(1).to_numpy()
            lo = dh.rolling(HOURS_60D).quantile(1 - q).shift(1).to_numpy()
            for side, thr, cmp in (("buy", hi, 1), ("sell", lo, -1)):
                idx = np.nonzero(np.isfinite(thr) & (dhv * cmp > thr * cmp))[0]
                for i in idx:
                    events.append({"stock": sym, "hour_ts": int(hour_end_ts[i]),
                                   "side": side, "q": q, "flow": float(dhv[i])})

        # 尖峰 sanity:日增幅>50% 事件,7 天后 shares/事件前 shares
        d_sh = s_sh.resample("1D").last().dropna()
        dp = d_sh.pct_change()
        for t in dp.index[dp > 0.5]:
            i = d_sh.index.get_loc(t)
            if i + 7 < len(d_sh):
                spike_check.append(d_sh.iloc[i + 7] / d_sh.iloc[i - 1])

    md = np.concatenate(all_min_diff)
    print(f"分钟差分(非零): 中位 |Δ|={np.median(np.abs(md)):,.0f} 股,"
          f" p99={np.percentile(np.abs(md),99):,.0f}, p99.99={np.percentile(np.abs(md),99.99):,.0f}")
    print(f"+50% 日尖峰 {len(spike_check)} 个:7 日后 shares/事件前 = "
          f"中位 {np.median(spike_check):.2f}(≈1=真建仓;≈2=增发后未回落... 看分布)")
    print(f"  尖峰后回落(<1.2)占比 {np.mean(np.array(spike_check)<1.2):.0%},"
          f" 维持(≥1.2)占比 {np.mean(np.array(spike_check)>=1.2):.0%}")

    ev = pd.DataFrame(events)
    print(f"巨鲸事件数: " + ", ".join(
        f"q={q}:买 {len(ev[(ev.q==q)&(ev.side=='buy')])}/卖 {len(ev[(ev.q==q)&(ev.side=='sell')])}"
        for q in Q_LIST))

    # z_20 面板(前一日收盘,可交易)
    close = pd.DataFrame(daily_close).ffill()
    z20 = zscore_panel(close).shift(1)       # shift:事件日的分类用前一日 z
    z20_by_day = {d: z20.loc[d] for d in z20.index}

    # ---------- 第二步:事件研究 ----------
    print("\n=== 第二步:事件研究(入场=事件小时收盘后第一分钟,无未来函数)===")
    px_map, ts_map = {}, {}
    for sym in stocks:
        ts, px, _ = load_minute(sym)
        ts_map[sym], px_map[sym] = ts, px

    def fwd_returns(sym, hour_ts):
        """从事件后可交易的第一分钟起,各 horizon 的远期收益。"""
        ts, px = ts_map[sym], px_map[sym]
        j = np.searchsorted(ts, hour_ts + 60)
        if j >= len(ts):
            return None, None
        e_px = px[j]
        out = {}
        for H, sec in HORIZONS.items():
            k = np.searchsorted(ts, ts[j] + sec)
            out[H] = (px[min(k, len(px) - 1)] / e_px - 1) if k < len(ts) else np.nan
        return e_px, out

    # 无条件均值:每小时末抽样
    uncond = {H: [] for H in HORIZONS}
    rng = np.random.default_rng(0)
    for sym in stocks:
        ts = ts_map[sym]
        hours = np.arange(ts[0] + 3600, ts[-1] - 15 * 86400, 3600)
        for h in hours[::6]:                 # 每 6 小时抽一个,控制计算量
            _, fr = fwd_returns(sym, h)
            if fr:
                for H in HORIZONS:
                    if np.isfinite(fr[H]):
                        uncond[H].append(fr[H])
    uncond_mean = {H: float(np.mean(v)) for H, v in uncond.items()}

    ev_rows = []
    for r in events:
        e_px, fr = fwd_returns(r["stock"], r["hour_ts"])
        if not fr:
            continue
        day = pd.Timestamp(r["hour_ts"], unit="s", tz="UTC").floor("D")
        zrow = z20_by_day.get(day)
        z = float(zrow[r["stock"]]) if zrow is not None else np.nan
        ev_rows.append({**r, "z": z, **{f"r_{H}": fr[H] for H in HORIZONS}})
    evr = pd.DataFrame(ev_rows)
    evr.to_csv(OUTPUT_DIR / "18_events.csv", index=False)

    study = []
    print(f"{'分组':<24}" + "".join(f"{H:>13}" for H in HORIZONS))
    for side in ("buy", "sell"):
        for zgrp, cond in (("all", evr.side == side),
                           ("z<-1", (evr.side == side) & (evr.z < -1)),
                           ("z>0", (evr.side == side) & (evr.z > 0))):
            sub = evr[cond & (evr.q == 0.995)]
            line = f"{side}/{zgrp}(n={len(sub)})".ljust(24)
            row = {"side": side, "zgrp": zgrp, "n": len(sub)}
            for H in HORIZONS:
                v = sub[f"r_{H}"].dropna()
                t = (v.mean() - uncond_mean[H]) / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 5 else np.nan
                row[f"{H}_bp"] = v.mean() * 1e4
                row[f"{H}_t"] = t
                line += f"{v.mean()*1e4:+6.0f}(t{t:+5.1f})".rjust(13)
            study.append(row)
            print(line)
    study_df = pd.DataFrame(study)
    print("  (括号内为 vs 无条件均值的 t 值;q=0.995)")

    # ---------- 控制 z 的增量(OLS) ----------
    print("\n=== 控制 z 后的巨鲸增量(7d 远期收益)===")
    day_z = z20.stack()
    day_ret7 = (close.shift(-7) / close - 1).stack()
    whale_days = set(zip(pd.to_datetime(evr[(evr.side == "buy") & (evr.q == 0.995)].hour_ts,
                                        unit="s", utc=True).dt.floor("D"),
                         evr[(evr.side == "buy") & (evr.q == 0.995)].stock))
    panel = pd.DataFrame({"z": day_z, "r7": day_ret7}).dropna()
    panel["whale"] = [1.0 if (d, s) in whale_days else 0.0
                      for d, s in panel.index]
    X = np.column_stack([np.ones(len(panel)), panel["z"], panel["whale"]])
    y = panel["r7"].to_numpy()
    beta, res_, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - 3)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    print(f"  r_7d = {beta[0]*1e4:+.1f}bp + ({beta[1]*1e4:+.2f}bp)·z + "
          f"({beta[2]*1e4:+.1f}bp)·whale_buy   (t_whale = {beta[2]/se[2]:+.1f},"
          f" n={len(y)}, whale 日数={int(panel.whale.sum())})")

    # ---------- 第三步:跟单回测 ----------
    print("\n=== 第三步:巨鲸跟单回测(P1 单仓位)===")
    n_days = len(close)
    dates = close.index
    split_ts = dates[int(n_days * 0.7)].timestamp()
    years_total = (dates[-1] - dates[0]).days / 365.25

    exits = ([("hold", h) for h in (1, 3, 7, 14)] +
             [("tp", tp, T) for tp in (0.005, 0.01) for T in (7, 14)])

    def run_strategy(q, zfilter, exit_rule):
        evs = evr[(evr.q == q) & (evr.side == "buy")].sort_values("hour_ts")
        if zfilter:
            evs = evs[evs.z < 0]
        trades = []          # (entry_ts, exit_ts, net)
        flat_from = 0.0
        for r in evs.itertuples():
            if r.hour_ts < flat_from:
                continue
            ts, px = ts_map[r.stock], px_map[r.stock]
            j = np.searchsorted(ts, r.hour_ts + 60)
            if j >= len(ts):
                continue
            e_ts, e_px = ts[j], px[j]
            kind = exit_rule[0]
            if kind == "hold":
                sec = exit_rule[1] * 86400
                k = np.searchsorted(ts, e_ts + sec)
                if k >= len(ts):
                    continue
                s_px = px[k]
            else:            # tp + 超时(分钟扫)
                _, tp, T = exit_rule
                target = e_px * (1 + tp)
                k_end = np.searchsorted(ts, e_ts + T * 86400)
                window = px[j:k_end]
                hit = np.nonzero(window >= target)[0]
                s_px = float(target) if len(hit) else px[min(k_end, len(px) - 1)]
            net = s_px * KEEP / e_px - 1
            trades.append((e_ts, 0, net))
            flat_from = e_ts + (exit_rule[1] if kind == "hold" else exit_rule[2]) * 86400
        # 权益:按成交时间复利,逐日平坦
        tr = [t for t in trades if t[0] < split_ts]
        te = [t for t in trades if t[0] >= split_ts]
        def ann(x):
            if not x:
                return np.nan
            tot = np.prod([1 + t[2] for t in x])
            yrs = years_total * (0.7 if x is tr else 0.3)
            return (tot ** (1 / yrs) - 1) * 100
        # maxDD:按交易序列
        eq = np.cumprod([1 + t[2] for t in trades]) if trades else np.array([1.0])
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1).min()) if len(eq) else np.nan
        nets = [t[2] for t in trades]
        return {"cagr_train": ann(tr), "cagr_test": ann(te), "n_trades": len(trades),
                "win_rate": float(np.mean([x > 0 for x in nets])) if nets else np.nan,
                "avg_net_bp": float(np.mean(nets)) * 1e4 if nets else np.nan,
                "max_dd": mdd}

    srows = []
    for q in Q_LIST:
        for zf in (False, True):
            for er in exits:
                m = run_strategy(q, zf, er)
                m.update(q=q, zfilter=zf, exit=str(er))
                srows.append(m)
    sres = pd.DataFrame(srows)
    sres.to_csv(OUTPUT_DIR / "18_whale_follow.csv", index=False)
    pd.set_option("display.width", 200)
    print(sres.round(3).to_string(index=False))
    ok = sres.dropna(subset=["cagr_train"])
    print(f"\ntrain/test 相关: {ok[['cagr_train','cagr_test']].corr().iloc[0,1]:.3f}")
    print(f"按 train 选优 Top-5 的 test: 中位 {ok.nlargest(5,'cagr_train').cagr_test.median():.1f}%")
    print(f"全部配置 test 中位: {ok.cagr_test.median():.1f}%,最高 {ok.cagr_test.max():.1f}%")

    # ---------- 双触发组合:巨鲸跟单 + 16 号 mid z 配置 ----------
    print("\n=== 组合:跟单(q=0.995, z<0, tp1%/7d) + z 超跌(W20 k1.0 tp0.6% T5 δ0.15%) ===")
    combo = run_combo(ev, evr, ts_map, px_map, close, z20, stocks, split_ts, years_total)
    print(f"  组合: train {combo['cagr_train']:.1f}% / test {combo['cagr_test']:.1f}%  "
          f"({combo['n_whale']} 笔跟单 + {combo['n_z']} 笔 z, maxDD {combo['max_dd']:.1%})")

    # ---------- 排序权重实验:whale 只做候选排序权重(mid 配置) ----------
    print("\n=== 排序权重实验:mid W20 k1.0 tp0.6% T5 P2 δ0.15% N1 ε0.3%,排序键 z - w×whale ===")
    z20_raw = zscore_panel(close).to_numpy()          # 不 shift:收盘 t 的 z 在收盘时已知
    px_d = close.to_numpy()
    ns_ = len(stocks)
    day_start = np.array([d.timestamp() for d in dates])
    day_end = day_start + 86400
    ev_buy = evr[(evr.q == 0.995) & (evr.side == "buy")]
    whale_mat = np.zeros((n_days, ns_))
    for si, sym in enumerate(stocks):
        ets_ = np.sort(ev_buy[ev_buy.stock == sym].hour_ts.to_numpy())
        whale_mat[:, si] = (np.searchsorted(ets_, day_end)
                            - np.searchsorted(ets_, day_end - 86400)) > 0

    def sim_weight(w, k=1.0, tp=0.006, T=5, P=2, delta=0.0015, eps=0.003):
        cash, pos, pend = 1.0, {}, []
        equity = np.empty(n_days)
        rets, holds = [], []
        for i in range(n_days - 1):
            d0, d1 = day_start[i], day_start[i + 1]
            for c_, p in list(pos.items()):
                ts, px = ts_map[stocks[c_]], px_map[stocks[c_]]
                w_ = px[np.searchsorted(ts, d0):np.searchsorted(ts, d1)]
                tgt = p["entry_px"] * (1 + tp) * (1 + eps)
                sell = float(tgt) if len(w_) and (w_ >= tgt).any() else None
                held = i - p["entry_i"] + 1
                if sell is None and held >= T and len(w_):
                    sell = float(w_[-1])
                if sell is not None:
                    cash += p["shares"] * sell * KEEP
                    rets.append(sell * KEEP / p["entry_px"] - 1)
                    holds.append(held)
                    del pos[c_]
            fills, still = [], []
            for c_, limit, left in pend:
                ts, px = ts_map[stocks[c_]], px_map[stocks[c_]]
                w_ = px[np.searchsorted(ts, d0):np.searchsorted(ts, d1)]
                fill = None
                if len(w_):
                    if (w_ <= limit).any():
                        fill = float(limit)
                    elif left <= 1:
                        fill = float(w_[-1])
                if fill is not None and c_ not in pos:
                    fills.append((c_, fill))
                elif fill is None:
                    still.append((c_, limit, left - 1))
            pend = still
            for j, (c_, fill) in enumerate(fills):
                if cash <= 1e-12:
                    break
                alloc = cash / (len(fills) - j)
                pos[c_] = {"shares": alloc / fill, "entry_px": fill, "entry_i": i}
                cash -= alloc
            equity[i] = cash + sum(p["shares"] * px_d[i][c_] for c_, p in pos.items())
            zr = z20_raw[i]
            if not np.isfinite(zr).any():
                continue
            slots = P - len(pos) - len(pend)
            if slots <= 0:
                continue
            cand = [c_ for c_ in range(ns_) if np.isfinite(zr[c_]) and zr[c_] < -k
                    and c_ not in pos and all(c_ != b[0] for b in pend)]
            cand.sort(key=lambda c_: zr[c_] - w * whale_mat[i][c_])
            for c_ in cand[:slots]:
                pend.append((c_, px_d[i][c_] * (1 - delta), 1))
        equity[n_days - 1] = cash + sum(p["shares"] * px_d[n_days - 1][c_] for c_, p in pos.items())
        eqs = pd.Series(equity[WARMUP:], index=dates[WARMUP:])
        si_ = int(n_days * 0.7)
        tr, te = eqs.iloc[:si_ - WARMUP], eqs.iloc[si_ - WARMUP:]
        eq = np.cumprod(1 + np.array(rets)) if rets else np.array([1.0])
        peak = np.maximum.accumulate(eq)
        return {"w": w,
                "cagr_train": cagr(tr / tr.iloc[0]) * 100,
                "cagr_test": cagr(te / te.iloc[0]) * 100,
                "n_trades": len(rets),
                "win_rate": float(np.mean([r > 0 for r in rets])) if rets else np.nan,
                "avg_net_bp": float(np.mean(rets)) * 1e4 if rets else np.nan,
                "max_dd": float((eq / peak - 1).min())}

    wrows = [sim_weight(w) for w in (0.0, 0.3, 0.65, 1.0)]
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(OUTPUT_DIR / "18_whale_weight.csv", index=False)
    print(wdf.round(3).to_string(index=False))

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for side, zgrp, mk in (("buy", "z<-1", "o-"), ("buy", "z>0", "s-"), ("sell", "all", "^--")):
        row = study_df[(study_df.side == side) & (study_df.zgrp == zgrp)]
        if len(row):
            ax.plot(range(len(HORIZONS)), [row.iloc[0][f"{H}_bp"] for H in HORIZONS],
                    mk, label=f"{side} {zgrp}")
    ax.plot(range(len(HORIZONS)), [uncond_mean[H] * 1e4 for H in HORIZONS],
            ":", c="gray", label="unconditional")
    ax.set_xticks(range(len(HORIZONS)), HORIZONS.keys())
    ax.set_ylabel("mean forward return (bp)")
    ax.set_title("Whale event study (q=0.995)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    axes[1].bar(["whale\nbest", "whale\nmedian", "combined"], [
        ok.cagr_test.max(),
        ok.cagr_test.median(),
        combo["cagr_test"]])
    axes[1].axhline(30.3, ls="--", c="gray", label="script-17 final test 30.3%")
    axes[1].set_ylabel("test CAGR (%)")
    axes[1].set_title("Strategy test CAGR vs baseline")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "18_whale_follow.png", dpi=150)
    print(f"\n输出: {OUTPUT_DIR / '18_whale_follow.csv'} / 18_events.csv / 18_whale_follow.png")


def run_combo(ev, evr, ts_map, px_map, close, z20, stocks, split_ts, years_total):
    """P1 单仓位双触发:空仓时,最近的巨鲸事件(下一分钟市价)或 z 信号
    (收盘判定,次日限价 close×(1-0.0015),当日未成交则市价)先进;
    巨鲸仓位 tp=1%/超时7d,z 仓位 tp=0.6%+ε0.3%/超时5d。"""
    evs = evr[(evr.q == 0.995) & (evr.side == "buy") & (evr.z < 0)].sort_values("hour_ts")
    ev_ts = evs.hour_ts.to_numpy()
    ev_sym = evs.stock.to_numpy()
    dates = close.index
    z = z20.to_numpy()
    px_d = close.to_numpy()
    n, ns = px_d.shape
    day_start = np.array([d.timestamp() for d in dates])

    trades = []              # (entry_ts, net, kind)
    pos = None               # dict(sym, entry_px, entry_ts, kind)
    pend = None              # (col, limit) z 限价单
    cash_used = 0.0

    def scan_exit(sym, e_px, e_ts, tp, T, cur_ts):
        ts, px = ts_map[sym], px_map[sym]
        j = np.searchsorted(ts, e_ts)
        k_end = np.searchsorted(ts, e_ts + T * 86400)
        w = px[j:k_end]
        hit = np.nonzero(w >= e_px * (1 + tp))[0]
        if len(hit):
            return float(e_px * (1 + tp))
        return float(px[min(k_end, len(px) - 1)])

    for i in range(n - 1):
        d0, d1 = day_start[i], day_start[i + 1]
        # 1) 持仓:简化——巨鲸/z 仓位都在信号次日按各自规则一次性回放退出
        #    (逐日盯市意义不大,直接按成交规则结算)
        # 2) 空仓:找本日内最早的巨鲸事件
        if pos is None and pend is None:
            m = np.searchsorted(ev_ts, d0)
            if m < len(ev_ts) and ev_ts[m] < d1 and ev_ts[m] >= cash_used:
                sym = ev_sym[m]
                ts, px = ts_map[sym], px_map[sym]
                j = np.searchsorted(ts, ev_ts[m] + 60)
                if j < len(ts):
                    s_px = scan_exit(sym, px[j], ts[j], 0.01, 7, d1)
                    net = s_px * KEEP / px[j] - 1
                    trades.append((ts[j], net, "whale"))
                    cash_used = ts[j] + 7 * 86400
                    continue
        # 3) z 限价单成交判定(空仓且无事件)
        if pos is None and pend is not None:
            col, limit = pend
            sym = stocks[col]
            ts, px = ts_map[sym], px_map[sym]
            j0 = np.searchsorted(ts, d0)
            j1 = np.searchsorted(ts, d1)
            w = px[j0:j1]
            hit = np.nonzero(w <= limit)[0]
            fill = float(limit) if len(hit) else (float(w[-1]) if len(w) else None)
            pend = None
            if fill is not None:
                ts_j = ts[min(j0 + (hit[0] if len(hit) else 0), len(ts) - 1)]
                k_end = np.searchsorted(ts, ts_j + 5 * 86400)
                w2 = px[np.searchsorted(ts, ts_j):k_end]
                tgt = fill * 1.006 * 1.003           # tp=0.6% × (1+ε),ε=0.3%
                hit2 = np.nonzero(w2 >= tgt)[0]
                s_px = float(tgt) if len(hit2) else float(px[min(k_end, len(px) - 1)])
                net = s_px * KEEP / fill - 1
                trades.append((ts_j, net, "z"))
                cash_used = ts_j + 5 * 86400
            continue
        # 4) 收盘 z 信号
        if pos is None and pend is None and cash_used <= d1:
            zr = z[i]
            cand = [c for c in range(ns) if np.isfinite(zr[c]) and zr[c] < -1.0]
            if cand:
                best = min(cand, key=lambda c: zr[c])
                pend = (best, px_d[i][best] * (1 - 0.0015))

    nets = np.array([t[1] for t in trades])
    ets = np.array([t[0] for t in trades]) if trades else np.array([])
    def ann(mask, yrs):
        if mask.sum() == 0:
            return np.nan
        return ((np.prod(1 + nets[mask])) ** (1 / yrs) - 1) * 100
    eq = np.cumprod(1 + nets) if len(nets) else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    return {"cagr_train": ann(ets < split_ts, years_total * 0.7),
            "cagr_test": ann(ets >= split_ts, years_total * 0.3),
            "n_whale": int(sum(1 for t in trades if t[2] == "whale")),
            "n_z": int(sum(1 for t in trades if t[2] == "z")),
            "max_dd": float((eq / peak - 1).min())}


if __name__ == "__main__":
    main()
