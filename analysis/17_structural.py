"""17_structural.py — 三个结构性想法(非参数微调),目标检验 70-80%/年是否可达。

全程前 70% 天数 train / 后 30% test,卖出税 0.1%,严格无未来函数
(t 收盘决策,t+1 收盘成交;统计量只用 t 及以前数据)。

实验 A:始终满仓连续轮换。永远持有 P∈{1,3} 只 z_20 最低的股票,
  仅当候选股 z 比持仓股低 h∈{0.25,0.5,0.75} 才换股(滞后带控制换手税)。
  对照:等权买入持有、16 号触发式基座(W20 k1.0 tp0.6% T5 P2 日线)。
实验 B:波动过滤叠加。候选池限制在"过去 20 日平均 range 前 50%",跑 16 号最稳两配置:
  ①分批建仓 W20 k0.75 tp0.6% T10(日线);②分钟执行 W20 k1.0 tp0.6% T5 P2 δ0.15% N1 ε0.3%。
实验 C:股本增发事件研究。total_shares 日增幅 >50% 为事件,统计事件后 1/3/7/14 日
  累计收益(t 检验 vs 无条件均值)与事件前 14 日走势;若显著则把"事件后 X 天 z 加分"
  叠加到实验 A 测增量,否则如实报告。

输出 analysis/output/17_structural.csv / 17_structural.png / 17_events.csv
用法: cd analysis && ../.venv/bin/python 17_structural.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUT_DIR, SELL_TAX, cagr, list_stocks, load_stock, max_drawdown

WARMUP = 32
KEEP = 1.0 - SELL_TAX


# ---------------------------------------------------------------- 数据
def load_all():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    o, h, l, c, sh = {}, {}, {}, {}, {}
    minute = {}
    for s in stocks:
        df = load_stock(s)
        day = df.index.floor("D")
        g = df.groupby(day)["price"]
        o[s], h[s], l[s], c[s] = g.first(), g.max(), g.min(), g.last()
        sh[s] = df.groupby(day)["total_shares"].last()
        minute[s] = {d: v["price"].to_numpy(dtype=np.float32)
                     for d, v in df.groupby(day)}
    mats = {k: pd.DataFrame(v) for k, v in
            dict(open=o, high=h, low=l, close=c, shares=sh).items()}
    return stocks, mats, minute


def zscore(close: pd.DataFrame, W: int) -> pd.DataFrame:
    return (close - close.rolling(W).mean()) / close.rolling(W).std()


def metrics(equity, dates, split_i, rets=None, holds=None, n_sells=None):
    eqs = pd.Series(equity[WARMUP:], index=dates[WARMUP:])
    out = {}
    for name, (a, b) in {"train": (0, split_i), "test": (split_i, len(dates))}.items():
        sub = eqs.iloc[max(0, a - WARMUP): b - WARMUP]
        out[f"cagr_{name}"] = cagr(sub / sub.iloc[0]) * 100 if len(sub) > 30 else np.nan
    out["max_dd"] = max_drawdown(eqs)
    if n_sells is not None:
        years = (dates[-1] - dates[WARMUP]).days / 365.25
        out["sells_per_year"] = n_sells / years
    if rets is not None:
        out["n_trades"] = len(rets)
        out["win_rate"] = float(np.mean([r > 0 for r in rets])) if rets else np.nan
        out["avg_net_bp"] = float(np.mean(rets)) * 1e4 if rets else np.nan
        out["avg_hold_d"] = float(np.mean(holds)) if holds else np.nan
    return out


# ---------------------------------------------------------------- 实验 A:连续轮换
def run_rotation(px, z, P, h, pool=None, tilt=None):
    """带纯买入分支的完整版(simulate_rotation 的包装,逻辑合一)。"""
    n, ns = px.shape
    zz = z if tilt is None else z + tilt
    cash = 1.0
    pos = {}
    pend = []                   # (sell_col or -1, buy_col)
    equity = np.empty(n)
    n_sells = 0

    for i in range(n):
        row = px[i]
        buys = []
        for sc, bc in pend:
            if sc >= 0 and sc in pos and np.isfinite(row[sc]):
                cash += pos[sc] * row[sc] * KEEP
                del pos[sc]
                n_sells += 1
            if np.isfinite(row[bc]):
                buys.append(bc)
        for j, bc in enumerate(buys):
            if cash <= 1e-12:
                break
            alloc = cash / (len(buys) - j)      # 剩余现金均分
            pos[bc] = pos.get(bc, 0) + alloc / row[bc]
            cash -= alloc
        pend = []
        equity[i] = cash + sum(s * row[c] for c, s in pos.items())

        zr = zz[i]
        if not np.isfinite(zr).any() or i >= n - 1:
            continue
        if len(pos) < P:
            cand = [c for c in np.argsort(zr)
                    if np.isfinite(zr[c]) and c not in pos
                    and (pool is None or pool[i][c])]
            for c in cand[:P - len(pos)]:
                pend.append((-1, c))
            continue
        decision = set(pos)
        for _ in range(P):
            worst = max(decision, key=lambda c: zr[c] if np.isfinite(zr[c]) else -np.inf)
            cand = [c for c in range(ns) if c not in decision and np.isfinite(zr[c])
                    and (pool is None or pool[i][c])]
            if not cand or not np.isfinite(zr[worst]):
                break
            best = min(cand, key=lambda c: zr[c])
            if zr[best] < zr[worst] - h:
                pend.append((worst, best))
                decision.discard(worst)
                decision.add(best)
            else:
                break
    return equity, n_sells


# ---------------------------------------------------------------- 实验 B 引擎(移植 16 号,加候选池)
def simulate_daily(feats, W, k, tp, T, P, scale_in=False, pool=None):
    z = feats[f"z{W}"]
    px = feats["close"]
    n, ns = px.shape
    cash = 1.0
    pos = {}
    pend_buy, pend_sell, pend_scale = [], [], []
    equity = np.empty(n)
    rets, holds = [], []

    for i in range(n):
        row = px[i]
        for c_ in pend_sell:
            if c_ in pos:
                cash += pos[c_]["shares"] * row[c_] * KEEP
                rets.append(pos[c_]["shares"] * row[c_] * KEEP / pos[c_]["invested"] - 1)
                holds.append(i - pos[c_]["entry_i"])
                del pos[c_]
        for c_ in pend_scale:
            if c_ in pos and not pos[c_]["scaled"] and np.isfinite(row[c_]) and cash > 1e-12:
                add = min(pos[c_]["invested"], cash)
                pos[c_]["shares"] += add / row[c_]
                pos[c_]["invested"] += add
                pos[c_]["entry_px"] = pos[c_]["invested"] / pos[c_]["shares"]
                pos[c_]["scaled"] = True
                cash -= add
        n_slots = min(P - len(pos), len(pend_buy))
        for j, c_ in enumerate(pend_buy[:n_slots]):
            if c_ not in pos and cash > 1e-12 and np.isfinite(row[c_]):
                alloc = cash / (n_slots - j)
                if scale_in:
                    alloc /= 2
                pos[c_] = {"shares": alloc / row[c_], "invested": alloc,
                           "entry_px": row[c_], "entry_i": i, "scaled": not scale_in}
                cash -= alloc
        pend_buy, pend_sell, pend_scale = [], [], []
        equity[i] = cash + sum(p["shares"] * row[c_] for c_, p in pos.items())

        zr = z[i]
        if not np.isfinite(zr).any():
            continue
        for c_, p in list(pos.items()):
            held = i - p["entry_i"]
            if row[c_] >= p["entry_px"] * (1 + tp) or held >= T:
                pend_sell.append(c_)
                continue
            if scale_in and not p["scaled"] and np.isfinite(zr[c_]) and zr[c_] < -(k + 0.75):
                pend_scale.append(c_)
        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = [c_ for c_ in range(ns)
                    if np.isfinite(zr[c_]) and zr[c_] < -k and c_ not in pos
                    and (pool is None or pool[i][c_])]
            cand.sort(key=lambda c_: zr[c_])
            pend_buy = cand[:slots]
    return equity, rets, holds


def simulate_minute(feats, dates, minute, stocks, W, k, tp, T, P,
                    delta, n_entry, eps, pool=None):
    z = feats[f"z{W}"]
    px = feats["close"]
    n, ns = px.shape
    day_list = list(dates)
    cash = 1.0
    pos = {}
    pend_buy = []
    equity = np.empty(n)
    rets, holds = [], []

    for i in range(n):
        row = px[i]
        for c_, p in list(pos.items()):
            arr = minute[stocks[c_]].get(day_list[i])
            sell = None
            target = p["entry_px"] * (1 + tp) * (1 + eps)
            if arr is not None and (arr >= target).any():
                sell = float(target)
            held = i - p["entry_i"] + 1
            if sell is None and held >= T and arr is not None:
                sell = float(arr[-1])
            if sell is not None:
                cash += p["shares"] * sell * KEEP
                rets.append(sell * KEEP / p["entry_px"] - 1)
                holds.append(held)
                del pos[c_]
        fills, still = [], []
        for c_, limit, left in pend_buy:
            arr = minute[stocks[c_]].get(day_list[i])
            fill = None
            if arr is not None:
                if (arr <= limit).any():
                    fill = float(limit)
                elif left <= 1:
                    fill = float(arr[-1])
            if fill is not None and c_ not in pos:
                fills.append((c_, fill))
            elif fill is None:
                still.append((c_, limit, left - 1))
        pend_buy = still
        for j, (c_, fill) in enumerate(fills):
            if cash <= 1e-12:
                break
            alloc = cash / (len(fills) - j)
            pos[c_] = {"shares": alloc / fill, "entry_px": fill, "entry_i": i}
            cash -= alloc
        equity[i] = cash + sum(p["shares"] * row[c_] for c_, p in pos.items())

        zr = z[i]
        if not np.isfinite(zr).any() or i >= n - 1:
            continue
        slots = P - len(pos) - len(pend_buy)
        if slots <= 0:
            continue
        cand = [c_ for c_ in range(ns)
                if np.isfinite(zr[c_]) and zr[c_] < -k
                and c_ not in pos and all(c_ != b[0] for b in pend_buy)
                and (pool is None or pool[i][c_])]
        cand.sort(key=lambda c_: zr[c_])
        for c_ in cand[:slots]:
            pend_buy.append((c_, row[c_] * (1 - delta), n_entry))
    return equity, rets, holds


# ---------------------------------------------------------------- 主流程
def main() -> None:
    stocks, mats, minute = load_all()
    dates = mats["close"].index
    n = len(dates)
    split_i = int(n * 0.7)
    close = mats["close"].ffill()
    px = close.to_numpy()
    print(f"区间 {dates[0].date()} → {dates[-1].date()} ({n} 天); "
          f"train 到 {dates[split_i].date()},test 从 {dates[split_i].date()} 起")

    feats = {"close": px}
    for W in (10, 20, 30):
        feats[f"z{W}"] = zscore(close, W).to_numpy()
    z20 = feats["z20"]
    rng20 = (mats["high"] / mats["low"] - 1).rolling(20).mean()
    # 波动池:预测 range 横截面前 50%
    pool = (rng20.rank(axis=1, pct=True) >= 0.5).to_numpy()

    rows = []

    # ---- 基线:等权买入持有 ----
    norm = px / px[WARMUP]
    ew_eq = np.nanmean(norm, axis=1)
    m = metrics(ew_eq, dates, split_i)
    m.update(exp="baseline_ew_buyhold", P=np.nan, h=np.nan)
    rows.append(m)
    print(f"[基线] 等权买入持有: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")

    # ---- 基线:16 号触发式 mid 配置(日线) ----
    eq, rets, holds = simulate_daily(feats, 20, 1.0, 0.006, 5, 2)
    m = metrics(eq, dates, split_i, rets, holds)
    m.update(exp="baseline_trigger_W20k10", P=2, h=np.nan)
    rows.append(m)
    print(f"[基线] 触发式 W20 k1.0 tp0.6% T5 P2: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")

    # ---- 实验 A:连续轮换 ----
    print("[A] 始终满仓连续轮换...")
    a_curves = {}
    for P in (1, 3):
        for h in (0.25, 0.5, 0.75):
            eq, nsells = run_rotation(px, z20, P, h)
            m = metrics(eq, dates, split_i, n_sells=nsells)
            m.update(exp=f"A_rotation_P{P}_h{h}", P=P, h=h)
            rows.append(m)
            a_curves[f"P{P} h{h}"] = eq
            print(f"  P{P} h{h}: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%  "
                  f"换手 {m['sells_per_year']:.0f} 次/年  maxDD {m['max_dd']:.1%}")

    # ---- 实验 B:波动过滤叠加 ----
    print("[B] 波动过滤(预测 range 前 50%)...")
    eq, rets, holds = simulate_daily(feats, 20, 0.75, 0.006, 10, 1, scale_in=True)
    m = metrics(eq, dates, split_i, rets, holds)
    m.update(exp="B_scalein_nofilter", P=1, h=np.nan)
    rows.append(m)
    print(f"  分批建仓 无过滤: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")
    eq, rets, holds = simulate_daily(feats, 20, 0.75, 0.006, 10, 1, scale_in=True, pool=pool)
    m = metrics(eq, dates, split_i, rets, holds)
    m.update(exp="B_scalein_volfilter", P=1, h=np.nan)
    rows.append(m)
    print(f"  分批建仓 +波动池: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")
    eq, rets, holds = simulate_minute(feats, dates, minute, stocks,
                                      20, 1.0, 0.006, 5, 2, 0.0015, 1, 0.003)
    m = metrics(eq, dates, split_i, rets, holds)
    m.update(exp="B_minute_nofilter", P=2, h=np.nan)
    rows.append(m)
    print(f"  分钟执行 无过滤: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")
    eq, rets, holds = simulate_minute(feats, dates, minute, stocks,
                                      20, 1.0, 0.006, 5, 2, 0.0015, 1, 0.003, pool=pool)
    m = metrics(eq, dates, split_i, rets, holds)
    m.update(exp="B_minute_volfilter", P=2, h=np.nan)
    rows.append(m)
    print(f"  分钟执行 +波动池: train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%")

    # ---- 实验 C:增发事件研究 ----
    print("[C] 股本增发事件研究...")
    dpct = mats["shares"].pct_change()
    events = (dpct > 0.5).to_numpy()          # (n, ns) bool
    ev_idx = np.argwhere(events)
    fwd = {1: [], 3: [], 7: [], 14: []}
    pre = []
    for i, c_ in ev_idx:
        if i + 14 >= n or i < 15:
            continue
        for H in fwd:
            fwd[H].append(px[i + H, c_] / px[i, c_] - 1)
        pre.append(px[i - 1, c_] / px[i - 14, c_] - 1)
    # 无条件均值:全样本 H 日收益
    ev_rows = []
    print(f"  事件数(日增幅>50%): {len(ev_idx)}")
    for H in (1, 3, 7, 14):
        all_r = (pd.DataFrame(px).shift(-H) / pd.DataFrame(px) - 1).to_numpy()
        uncond = np.nanmean(all_r[WARMUP:-H])
        e = np.array(fwd[H])
        t = (e.mean() - uncond) / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 2 else np.nan
        ev_rows.append({"H": H, "n": len(e), "event_mean_bp": e.mean() * 1e4,
                        "uncond_bp": uncond * 1e4, "t_stat": t})
        print(f"  事件后 {H:2d} 日: 均值 {e.mean()*1e4:+7.1f}bp vs 无条件 {uncond*1e4:+5.1f}bp  (t={t:+.1f})")
    pre = np.array(pre)
    print(f"  事件前 14 日: 均值 {pre.mean()*1e4:+.0f}bp (t={pre.mean()/(pre.std(ddof=1)/np.sqrt(len(pre))):+.1f})")
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(OUTPUT_DIR / "17_events.csv", index=False)

    # 事件后漂移显著性判定 → 事件倾斜叠加到 A 最优配置
    sig_H = None
    for r in ev_rows:
        if r["t_stat"] > 2 and r["event_mean_bp"] > 0:
            sig_H = r["H"]
    if sig_H is not None:
        print(f"  事件后 {sig_H} 日漂移显著为正,叠加事件倾斜到 A...")
        tilt = np.zeros_like(z20)
        ns_ = px.shape[1]
        last_ev = np.full(ns_, -10**9)
        for i in range(n):
            hit = events[i]
            last_ev[hit] = i
            within = (i - last_ev) <= sig_H
            tilt[i][within] = -0.5      # 事件后 H 天内 z 减 0.5(提高持有优先级)
    else:
        print("  事件后漂移不显著 → 不做事件倾斜(如实报告:无事件结构)")
        tilt = None

    # ---- 最终叠加:A 最优(P,h 按 train 选)+ 波动池 + (可选)事件倾斜 ----
    a_rows = [r for r in rows if r["exp"].startswith("A_rotation")]
    best_a = max(a_rows, key=lambda r: r["cagr_train"])
    P_best, h_best = int(best_a["P"]), float(best_a["h"])
    print(f"\n[最终] A 最优(按 train): P{P_best} h{h_best}")
    # 实验 B 已证伪(波动过滤有害),最终策略只叠加有效杠杆:A + 事件倾斜,不加波动池
    eq, nsells = run_rotation(px, z20, P_best, h_best, tilt=tilt)
    m = metrics(eq, dates, split_i, n_sells=nsells)
    m.update(exp=f"FINAL_A_P{P_best}_h{h_best}_tilt{tilt is not None}",
             P=P_best, h=h_best)
    rows.append(m)
    print(f"[最终] 连续轮换 P{P_best} h{h_best} + 事件倾斜({tilt is not None},不加波动池): "
          f"train {m['cagr_train']:.1f}% / test {m['cagr_test']:.1f}%  "
          f"换手 {m['sells_per_year']:.0f} 次/年  maxDD {m['max_dd']:.1%}")
    final_eq = eq
    # 对照:再叠波动池(预期更差,验证 B 的结论)
    eq2, ns2 = run_rotation(px, z20, P_best, h_best, pool=pool, tilt=tilt)
    m2 = metrics(eq2, dates, split_i, n_sells=ns2)
    m2.update(exp=f"FINAL_A_P{P_best}_h{h_best}_pool_tilt{tilt is not None}",
              P=P_best, h=h_best)
    rows.append(m2)
    print(f"[对照] 再叠波动池: train {m2['cagr_train']:.1f}% / test {m2['cagr_test']:.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "17_structural.csv", index=False)

    # ---- 图:A 各配置净值 vs 基线 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    dts = dates[WARMUP:]
    for label, eq in a_curves.items():
        s = pd.Series(eq[WARMUP:], index=dts)
        ax.plot(dts, s / s.iloc[0], label=f"A {label}", lw=1)
    s = pd.Series(final_eq[WARMUP:], index=dts)
    ax.plot(dts, s / s.iloc[0], label="FINAL (A+pool+tilt)", lw=1.8, c="k")
    s = pd.Series(ew_eq[WARMUP:], index=dts)
    ax.plot(dts, s / s.iloc[0], label="equal-weight B&H", lw=1, ls="--", c="gray")
    ax.axvline(dates[split_i], c="r", ls=":", label="train/test split")
    ax.set_yscale("log")
    ax.set_title("Experiment A: always-in z_20 rotation vs baselines")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "17_structural.png", dpi=150)

    # ---- 70-80% 的算术 ----
    print("\n=== 70-80%/年 的算术对照 ===")
    for target in (0.70, 0.80):
        print(f"  {target:.0%}/年 需要日均净 {(1+target)**(1/365.25)-1:+.4%} "
              f"({((1+target)**(1/365.25)-1)*1e4:.1f}bp/天)")
    best_test = max(r["cagr_test"] for r in rows if not np.isnan(r["cagr_test"]))
    print(f"  本实验全部组合 test 最高 {best_test:.1f}%/年 "
          f"(日均 {((1+best_test/100)**(1/365.25)-1)*1e4:.1f}bp)")
    deep = (z20 < -1.5)
    print(f"  深度超跌供给:z_20<-1.5 平均 {deep[WARMUP:].sum()/((n-WARMUP)/365.25):.0f} 股·次/年;"
          f" z_20<-1.0 平均 {(z20[WARMUP:]<-1.0).sum()/((n-WARMUP)/365.25):.0f} 股·次/年")
    print(f"输出: {OUTPUT_DIR / '17_structural.csv'} / 17_events.csv / 17_structural.png")


if __name__ == "__main__":
    main()
