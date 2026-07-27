"""16_aggressive.py — 把 z 慢回复轮动从 25-27%/年往 70-80% 推:五个杠杆的系统测试。

全程严格无未来函数:信号用第 t 天收盘及以前的数据,成交在 t+1 或之后;
统一前 70% 天数 train / 后 30% 天数 test,两段分别报年化。

杠杆 1:更短持仓轮动网格。W∈{10,20,30} × k∈{0.75,1.0,1.25,1.5} ×
  止盈∈{0.4%,0.6%,1.0%} × 超时∈{3,5,7,10} × P∈{1,2,3},日线收盘执行(与 10 号工厂同语义)。
杠杆 2:分钟级执行优化。信号日收盘 close_t 后,买单限价 close_t×(1-δ),δ∈{0,.15%,.3%,.5%},
  N∈{1,2} 天内未触及则以第 N 天收盘市价买;止盈卖单限价 entry×(1+tp)×(1+ε),ε∈{0,.15%,.3%},
  超时日收盘市价卖。只对 4 个代表性日线参数组跑(不炸网格)。
杠杆 3:分批建仓(P1):z<-k 买 50%,持仓中 z<-(k+0.75) 再补 50%(次日收盘)。
杠杆 4:候选排序键:①z 最小;②dshares_5d>0 优先再 z 最小;③z 名次 + (-dshares) 名次之和。
杠杆 5:超跌子集 GBM 排序:特征 z_10/20/30、ret_5/10/20、dshares_5d、当日 range、
  5 日平均 range,预测未来 5 日收益,每 60 天 walk-forward 重训(标签窗口完整落在训练截止前)。

输出 analysis/output/16_aggressive.csv(全部组合)
用法: cd analysis && ../.venv/bin/python 16_aggressive.py
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import OUTPUT_DIR, SELL_TAX, cagr, list_stocks, load_stock, max_drawdown

W_LIST = (10, 20, 30)
K_LIST = (0.75, 1.0, 1.25, 1.5)
TP_LIST = (0.004, 0.006, 0.01)
T_LIST = (3, 5, 7, 10)
P_LIST = (1, 2, 3)
WARMUP = 32  # max(W)+2,与 10 号工厂一致

# 杠杆 2 的代表性日线参数组:(标签, W, k, tp, T, P)
L2_CONFIGS = [
    ("base_w30k15", 30, 1.5, 0.006, 7, 3),      # 最接近原 zflow 配置
    ("fast_w10k10", 10, 1.0, 0.004, 3, 1),      # 高换手原型
    ("mid_w20k10", 20, 1.0, 0.006, 5, 2),
    ("best_train", None, None, None, None, None),  # 跑完杠杆 1 后填 train 最优
]
DELTA_LIST = (0.0, 0.0015, 0.003, 0.005)
N_LIST = (1, 2)
EPS_LIST = (0.0, 0.0015, 0.003)


# ---------------------------------------------------------------- 数据准备
def load_all():
    """返回日频矩阵(dict of DataFrame)与分钟数据 {sym: {day: float32 分钟价数组}}。"""
    stocks = [s for s in list_stocks() if s != "TCSE"]
    o, h, l, c, sh = {}, {}, {}, {}, {}
    minute = {}
    for s in stocks:
        df = load_stock(s)
        day = df.index.floor("D")
        g = df.groupby(day)["price"]
        o[s] = g.first()
        h[s] = g.max()
        l[s] = g.min()
        c[s] = g.last()
        sh[s] = df.groupby(day)["total_shares"].last()
        minute[s] = {d: v["price"].to_numpy(dtype=np.float32)
                     for d, v in df.groupby(day)}
    mats = {k: pd.DataFrame(v) for k, v in
            dict(open=o, high=h, low=l, close=c, shares=sh).items()}
    return mats, minute


def build_features(mats):
    close = mats["close"].ffill()
    feats = {"close": close.to_numpy()}
    for W in W_LIST:
        ma, sd = close.rolling(W).mean(), close.rolling(W).std()
        feats[f"z{W}"] = ((close - ma) / sd).to_numpy()
    feats["dshares5"] = mats["shares"].pct_change(5).to_numpy()
    feats["range"] = (mats["high"] / mats["low"] - 1).to_numpy()
    for k in ("open", "high", "low"):
        feats[k] = mats[k].to_numpy()
    return feats


# ---------------------------------------------------------------- 日线回测引擎
def simulate_daily(feats, dates, W, k, tp, T, P, rank_key=1, scale_in=False,
                   rank_override=None):
    """日线收盘执行(10 号工厂语义):t 收盘出信号,t+1 收盘成交;
    退出判定在收盘,次日收盘卖出。rank_override: 每天的自定义排名分(小=优先,杠杆 5)。"""
    z = feats[f"z{W}"]
    flow = feats["dshares5"]
    px = feats["close"]
    n, ns = px.shape
    keep = 1.0 - SELL_TAX
    cash = 1.0
    pos = {}            # col -> dict(shares, invested, entry_px, entry_i, scaled)
    pend_buy, pend_sell, pend_scale = [], [], []
    equity = np.empty(n)
    rets, holds = [], []

    for i in range(n):
        row = px[i]
        for c_ in pend_sell:
            if c_ in pos:
                cash += pos[c_]["shares"] * row[c_] * keep
                rets.append(pos[c_]["shares"] * row[c_] * keep / pos[c_]["invested"] - 1)
                holds.append(i - pos[c_]["entry_i"])
                del pos[c_]
        for c_ in pend_scale:
            if c_ in pos and not pos[c_]["scaled"] and np.isfinite(row[c_]) and cash > 1e-12:
                add = min(pos[c_]["invested"], cash)     # 第二批与第一批等额
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
                    alloc /= 2                            # 首批半仓
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
                    if np.isfinite(zr[c_]) and zr[c_] < -k and c_ not in pos]
            if rank_override is not None:
                cand.sort(key=lambda c_: rank_override[i][c_])
            elif rank_key == 1:
                cand.sort(key=lambda c_: zr[c_])
            elif rank_key == 2:      # 净流入优先,再按 z
                cand.sort(key=lambda c_: (not (np.isfinite(flow[i][c_]) and flow[i][c_] > 0), zr[c_]))
            else:                    # z 名次 + 净流出名次
                zx = pd.Series(zr).rank().to_numpy()
                fx = pd.Series(-flow[i]).rank().to_numpy()
                cand.sort(key=lambda c_: zx[c_] + fx[c_])
            pend_buy = cand[:slots]

    return equity, rets, holds


# ---------------------------------------------------------------- 分钟执行引擎
def simulate_minute(feats, dates, minute, stocks, W, k, tp, T, P,
                    delta, n_entry, eps):
    """信号 t 收盘:z<-k 选股(z 最小优先);买单限价 close_t×(1-δ),t+1..t+N 内
    触及即成交(成交价=限价),否则第 N 天收盘市价买;卖单限价 entry×(1+tp)×(1+ε),
    分钟触及即卖;第 T 天(成交日记第 1 天)收盘市价强平。"""
    z = feats[f"z{W}"]
    px = feats["close"]
    n, ns = px.shape
    keep = 1.0 - SELL_TAX
    day_list = list(dates)

    def minutes(c_, d):
        return minute[stocks[c_]].get(d)

    cash = 1.0
    pos = {}            # col -> dict(shares, entry_px, entry_i)
    pend_buy = []       # (col, limit, days_left)
    equity = np.empty(n)
    rets, holds = [], []

    for i in range(n):
        row = px[i]
        # 1) 持仓:扫当日分钟找止盈;超时则收盘市价卖
        for c_, p in list(pos.items()):
            arr = minutes(c_, day_list[i])
            sell = None
            target = p["entry_px"] * (1 + tp) * (1 + eps)
            if arr is not None:
                hit = np.nonzero(arr >= target)[0]
                if len(hit):
                    sell = float(target)
            held = i - p["entry_i"] + 1
            if sell is None and held >= T and arr is not None:
                sell = float(arr[-1])
            if sell is not None:
                cash += p["shares"] * sell * keep
                rets.append(sell * keep / p["entry_px"] - 1)
                holds.append(held)
                del pos[c_]
        # 2) 待买:限价单当日有效;先确定全部成交,再均分现金
        fills, still = [], []
        for c_, limit, left in pend_buy:
            arr = minutes(c_, day_list[i])
            fill = None
            if arr is not None:
                if (arr <= limit).any():
                    fill = float(limit)
                elif left <= 1:
                    fill = float(arr[-1])          # 到期市价买
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

        # 3) 收盘出信号
        zr = z[i]
        if not np.isfinite(zr).any() or i >= n - 1:
            continue
        slots = P - len(pos) - len(pend_buy)
        if slots <= 0:
            continue
        cand = [c_ for c_ in range(ns)
                if np.isfinite(zr[c_]) and zr[c_] < -k
                and c_ not in pos and all(c_ != b[0] for b in pend_buy)]
        cand.sort(key=lambda c_: zr[c_])
        for c_ in cand[:slots]:
            pend_buy.append((c_, row[c_] * (1 - delta), n_entry))

    return equity, rets, holds


# ---------------------------------------------------------------- 指标
def metrics(equity, dates, rets, holds, split_i):
    eqs = pd.Series(equity[WARMUP:], index=dates[WARMUP:])
    out = {}
    for name, (a, b) in {"train": (0, split_i), "test": (split_i, len(dates))}.items():
        sub = eqs.iloc[max(0, a - WARMUP): b - WARMUP]
        out[f"cagr_{name}"] = cagr(sub / sub.iloc[0]) * 100 if len(sub) > 30 else np.nan
    out["max_dd"] = max_drawdown(eqs)
    out["n_trades"] = len(rets)
    out["win_rate"] = float(np.mean([r > 0 for r in rets])) if rets else np.nan
    out["avg_net_bp"] = float(np.mean(rets)) * 1e4 if rets else np.nan
    out["avg_hold_d"] = float(np.mean(holds)) if holds else np.nan
    return out


# ---------------------------------------------------------------- 杠杆 5:ML 排名
def ml_rank_scores(feats, dates, split_i):
    """每天给每只股票一个排名分(小=优先)。只在有候选的日子真正用;无预测的日子回退 z 名次。"""
    close = feats["close"]
    n, ns = close.shape
    X_all = np.column_stack([feats["z10"].ravel(), feats["z20"].ravel(), feats["z30"].ravel(),
                             feats["dshares5"].ravel(), feats["range"].ravel()])
    r5 = pd.DataFrame(close).pct_change(5).to_numpy().ravel()
    r20 = pd.DataFrame(close).pct_change(20).to_numpy().ravel()
    rng5 = pd.DataFrame(feats["range"]).rolling(5).mean().to_numpy().ravel()
    X_all = np.column_stack([X_all, r5, r20, rng5])
    y5 = (pd.DataFrame(close).shift(-5) / pd.DataFrame(close) - 1).to_numpy().ravel()
    day_idx = np.repeat(np.arange(n), ns)

    scores = np.full((n, ns), np.inf)
    model = None
    for i in range(n):
        if i % 60 == 0 and i > 200:
            tr = (day_idx <= i - 6) & np.isfinite(X_all).all(axis=1) & np.isfinite(y5)
            model = HistGradientBoostingRegressor(max_iter=150, random_state=0)
            model.fit(X_all[tr], y5[tr])
        if model is not None:
            row = X_all[i * ns:(i + 1) * ns]
            ok = np.isfinite(row).all(axis=1)
            pred = np.full(ns, np.inf)
            pred[ok] = -model.predict(row[ok])     # 预测收益高 → 分小 → 优先
            scores[i] = pred
    return scores


# ---------------------------------------------------------------- 主流程
def main() -> None:
    mats, minute = load_all()
    stocks = [s for s in list_stocks() if s != "TCSE"]
    feats = build_features(mats)
    dates = mats["close"].index
    n = len(dates)
    split_i = int(n * 0.7)
    print(f"区间 {dates[0].date()} → {dates[-1].date()} ({n} 天); "
          f"train 到 {dates[split_i].date()},test 从 {dates[split_i].date()} 起")

    rows = []

    # ---- 杠杆 1:日线网格 ----
    grid1 = list(itertools.product(W_LIST, K_LIST, TP_LIST, T_LIST, P_LIST))
    print(f"[杠杆1] 日线网格 {len(grid1)} 组...")
    for W, k, tp, T, P in grid1:
        eq, rets, holds = simulate_daily(feats, dates, W, k, tp, T, P)
        m = metrics(eq, dates, rets, holds, split_i)
        m.update(lever="L1_grid", W=W, k=k, tp=tp, T=T, P=P,
                 rank_key=1, scale_in=False, delta=np.nan, n_entry=np.nan, eps=np.nan)
        rows.append(m)

    # ---- 杠杆 3:分批建仓(P1) ----
    grid3 = list(itertools.product(W_LIST, K_LIST, TP_LIST, T_LIST))
    print(f"[杠杆3] 分批建仓 {len(grid3)} 组...")
    for W, k, tp, T in grid3:
        eq, rets, holds = simulate_daily(feats, dates, W, k, tp, T, 1, scale_in=True)
        m = metrics(eq, dates, rets, holds, split_i)
        m.update(lever="L3_scale_in", W=W, k=k, tp=tp, T=T, P=1,
                 rank_key=1, scale_in=True, delta=np.nan, n_entry=np.nan, eps=np.nan)
        rows.append(m)

    # ---- 杠杆 4:排序键(子网格) ----
    grid4 = list(itertools.product((20, 30), (1.0, 1.5), (0.006, 0.01), (5, 10), (2, 3)))
    print(f"[杠杆4] 排序键 {len(grid4)}×2 组...")
    for W, k, tp, T, P in grid4:
        for rk in (2, 3):
            eq, rets, holds = simulate_daily(feats, dates, W, k, tp, T, P, rank_key=rk)
            m = metrics(eq, dates, rets, holds, split_i)
            m.update(lever=f"L4_rank{rk}", W=W, k=k, tp=tp, T=T, P=P,
                     rank_key=rk, scale_in=False, delta=np.nan, n_entry=np.nan, eps=np.nan)
            rows.append(m)

    res = pd.DataFrame(rows)
    # 填杠杆 2 的 best_train 配置
    l1 = res[res.lever == "L1_grid"]
    best = l1.loc[l1.cagr_train.idxmax()]
    L2_CONFIGS[-1] = ("best_train", int(best["W"]), float(best["k"]), float(best["tp"]),
                      int(best["T"]), int(best["P"]))
    print(f"[杠杆1] train 最优: W{best["W"]} k{best["k"]} tp{best["tp"]} T{best["T"]} P{best["P"]} "
          f"train {best.cagr_train:.1f}% / test {best.cagr_test:.1f}%")

    # ---- 杠杆 2:分钟执行 ----
    print("[杠杆2] 分钟执行优化...")
    for label, W, k, tp, T, P in L2_CONFIGS:
        for delta, ne, eps in itertools.product(DELTA_LIST, N_LIST, EPS_LIST):
            eq, rets, holds = simulate_minute(feats, dates, minute, stocks,
                                              W, k, tp, T, P, delta, ne, eps)
            m = metrics(eq, dates, rets, holds, split_i)
            m.update(lever=f"L2_{label}", W=W, k=k, tp=tp, T=T, P=P,
                     rank_key=1, scale_in=False, delta=delta, n_entry=ne, eps=eps)
            rows.append(m)
    res = pd.DataFrame(rows)

    # ---- 杠杆 5:ML 排序(用 train 最优配置) ----
    print("[杠杆5] GBM 超跌排序(walk-forward 每 60 天重训)...")
    scores = ml_rank_scores(feats, dates, split_i)
    _, W, k, tp, T, P = L2_CONFIGS[-1]
    eq, rets, holds = simulate_daily(feats, dates, W, k, tp, T, P, rank_override=scores)
    m = metrics(eq, dates, rets, holds, split_i)
    m.update(lever="L5_ml_rank", W=W, k=k, tp=tp, T=T, P=P,
             rank_key=5, scale_in=False, delta=np.nan, n_entry=np.nan, eps=np.nan)
    rows.append(m)
    res = pd.DataFrame(rows)

    res.to_csv(OUTPUT_DIR / "16_aggressive.csv", index=False)
    print(f"\n总组合数 {len(res)},已写 {OUTPUT_DIR / "16_aggressive.csv"}")

    # ================ 汇总报告 ================
    pd.set_option("display.width", 250)
    cols = ["lever", "W", "k", "tp", "T", "P", "rank_key", "scale_in",
            "delta", "n_entry", "eps", "cagr_train", "cagr_test",
            "n_trades", "win_rate", "avg_net_bp", "avg_hold_d", "max_dd"]

    print("\n=== test 年化 Top-20(含运气成分,仅供对照)===")
    print(res.nlargest(20, "cagr_test")[cols].round(3).to_string(index=False))

    print("\n=== 诚实性检查:train Top-10 的 test 表现 ===")
    top10 = res.nlargest(10, "cagr_train")
    print(top10[cols].round(3).to_string(index=False))
    print(f"train Top-10 的 test 年化: 中位 {top10.cagr_test.median():.1f}%,"
          f"最大 {top10.cagr_test.max():.1f}%,最小 {top10.cagr_test.min():.1f}%")
    cc = res[["cagr_train", "cagr_test"]].corr().iloc[0, 1]
    print(f"全部组合 train/test 相关: {cc:.3f}")

    print("\n=== 杠杆 2 对照:同参数组 日线 vs 分钟执行(test 年化)===")
    for label, W, k, tp, T, P in L2_CONFIGS:
        base = res[(res.lever == "L1_grid") & (res["W"] == W) & (res["k"] == k) &
                   (res["tp"] == tp) & (res["T"] == T) & (res["P"] == P)]
        l2 = res[res.lever == f"L2_{label}"]
        plain = l2[(l2.delta == 0) & (l2.n_entry == 1) & (l2.eps == 0)]
        bestl2 = l2.loc[l2.cagr_train.idxmax()]
        print(f"  {label} (W{W} k{k} tp{tp} T{T} P{P}): "
              f"日线 {base.cagr_test.iloc[0] if len(base) else np.nan:.1f}% → "
              f"分钟δ0ε0 {plain.cagr_test.iloc[0] if len(plain) else np.nan:.1f}% → "
              f"分钟最优(δ={bestl2.delta},N={int(bestl2.n_entry)},ε={bestl2.eps},按train选) "
              f"train {bestl2.cagr_train:.1f}% / test {bestl2.cagr_test:.1f}%")

    print("\n=== 消融:逐个加杠杆(train 最优日线配置为基座)===")
    a0 = l1.loc[l1.cagr_train.idxmax()]
    print(f"  A0 日线基座 W{a0["W"]} k{a0["k"]} tp{a0["tp"]} T{a0["T"]} P{a0["P"]}: "
          f"train {a0.cagr_train:.1f}% / test {a0.cagr_test:.1f}%")
    l4b = res[res.lever.str.startswith("L4")].nlargest(1, "cagr_train")
    if len(l4b):
        r = l4b.iloc[0]
        print(f"  A1 +排序键{int(r.rank_key)} (W{r["W"]} k{r["k"]} tp{r["tp"]} T{r["T"]} P{r["P"]}): "
              f"train {r.cagr_train:.1f}% / test {r.cagr_test:.1f}%")
    l2b = res[res.lever == "L2_best_train"].loc[
        res[res.lever == "L2_best_train"].cagr_train.idxmax()]
    print(f"  A2 +分钟执行(δ={l2b.delta},N={int(l2b.n_entry)},ε={l2b.eps}): "
          f"train {l2b.cagr_train:.1f}% / test {l2b.cagr_test:.1f}%")
    l3b = res[res.lever == "L3_scale_in"].loc[
        res[res.lever == "L3_scale_in"].cagr_train.idxmax()]
    print(f"  A3 分批建仓最优(W{l3b["W"]} k{l3b["k"]} tp{l3b["tp"]} T{l3b["T"]}): "
          f"train {l3b.cagr_train:.1f}% / test {l3b.cagr_test:.1f}%")
    l5r = res[res.lever == "L5_ml_rank"].iloc[0]
    print(f"  A4 ML 排序(同基座参数): train {l5r.cagr_train:.1f}% / test {l5r.cagr_test:.1f}%")


if __name__ == "__main__":
    main()
