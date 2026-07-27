"""exp_grid_search.py — 系统化参数网格搜索: 找到原始策略未覆盖的最优参数。

原始报告只测了k={1.0, 1.5}、W=30。本实验扩展搜索范围:
- k ∈ {0.8, 1.0, 1.2, 1.5, 1.8, 2.0}
- W ∈ {20, 30, 40}
- P ∈ {2, 3, 4, 5}
- vol_filter ∈ {True, False}
- flow_filter ∈ {True, False}

总计288组配置,使用日线回测。

用法: .venv/bin/python analysis/exp_grid_search.py
"""

import itertools
import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_grid")
INIT_CAPITAL = 1.0


def load_panel():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        daily = resample_close(df, "1D")
        closes[sym] = daily
        shares[sym] = df["total_shares"].resample("1D").last().reindex(daily.index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def build_signals(px, sh, W):
    sig = {}
    sig["z"] = (px - px.rolling(W).mean()) / px.rolling(W).std()
    sig["flow"] = sh.pct_change(5)
    sig["vol_ratio"] = (px.pct_change().rolling(20).std() /
                        px.pct_change().rolling(200).std().expanding().median().replace(0, np.nan))
    return sig


def simulate(px, sig, params):
    dates = px.index
    P = params["P"]
    k = params["k"]
    T = params.get("T", 90)
    use_flow = params.get("use_flow", False)
    use_vol_filter = params.get("use_vol_filter", False)

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    warmup = max(100, params.get("W", 30) * 2)
    start_i = warmup

    for i in range(start_i, len(dates)):
        today = px.iloc[i]
        today_z = sig["z"].iloc[i]
        today_vol = sig["vol_ratio"].iloc[i]

        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym, "entry_date": dates[pos[sym]["entry_i"]],
                    "exit_date": dates[i],
                    "net_ret": today[sym] * (1 - SELL_TAX) / pos[sym]["entry_px"] - 1,
                    "hold_days": i - pos[sym]["entry_i"],
                })
                del pos[sym]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                alloc = cash / (n_slots - j)
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i}
                cash -= alloc
        pend_sell, pend_buy = [], []

        eq = cash + sum(p["shares"] * today[s] for s, p in pos.items())
        equity.append(eq)

        held = list(pos.keys())
        for sym in held:
            p = pos[sym]
            held_days = i - p["entry_i"]
            z_val = today_z[sym]
            exit_now = not np.isnan(z_val) and z_val >= 0
            if exit_now or held_days >= T:
                pend_sell.append(sym)

        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = today_z.dropna()
            cand = cand[(cand < -k) & ~cand.index.isin(pos)]
            if use_flow:
                fl = sig["flow"].iloc[i]
                cand = cand[fl.reindex(cand.index) > 0]
            if use_vol_filter:
                for sym in list(cand.index):
                    vr = today_vol[sym]
                    if not pd.isna(vr) and vr > 1.5:
                        if cand[sym] > -(k + 0.5):
                            cand = cand.drop(sym)
            ranked = cand.sort_values()
            pend_buy = list(ranked.index[:slots])

    eq = pd.Series(equity, index=dates[start_i:])
    return eq, trades


def run_config(px, sig, params, label):
    eq, trades = simulate(px, sig, params)
    if len(eq) < 30:
        return []
    rows = []
    spans = {"train": (eq.index[0], SPLIT_TRAIN_END),
             "valid": (SPLIT_TRAIN_END, SPLIT_VALID_END),
             "test": (SPLIT_VALID_END, eq.index[-1])}
    td = pd.DataFrame(trades)
    for name, (a, b) in spans.items():
        sub = eq[(eq.index >= a) & (eq.index < b)]
        if len(sub) < 30:
            continue
        sub = sub / sub.iloc[0]
        t = td[(td.exit_date >= a) & (td.exit_date < b)] if len(td) else td
        m = {"config": label, "split": name, "cagr": cagr(sub),
             "max_dd": max_drawdown(sub),
             "sharpe": ann_sharpe(sub.pct_change().dropna(), 365.25),
             "n_trades": len(t)}
        if len(t):
            m.update({"win_rate": (t.net_ret > 0).mean(),
                      "avg_net_bp": t.net_ret.mean() * 1e4,
                      "avg_hold_d": t.hold_days.mean()})
        rows.append(m)
    full_eq = eq / eq.iloc[0]
    full_m = {"config": label, "split": "full", "cagr": cagr(full_eq),
              "max_dd": max_drawdown(full_eq),
              "sharpe": ann_sharpe(full_eq.pct_change().dropna(), 365.25),
              "n_trades": len(td)}
    if len(td):
        full_m.update({"win_rate": (td.net_ret > 0).mean(),
                       "avg_net_bp": td.net_ret.mean() * 1e4,
                       "avg_hold_d": td.hold_days.mean()})
    rows.append(full_m)
    return rows


def main():
    px, sh = load_panel()

    k_vals = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    w_vals = [20, 30, 40]
    p_vals = [2, 3, 4, 5]
    flow_vals = [False, True]
    vol_vals = [False, True]

    total = len(k_vals) * len(w_vals) * len(p_vals) * len(flow_vals) * len(vol_vals)
    print(f"共 {total} 组配置, 开始搜索...")

    # 预计算不同W的信号
    sig_cache = {}
    for W in w_vals:
        sig_cache[W] = build_signals(px, sh, W)

    all_rows = []
    bh = (px / px.iloc[0]).mean(axis=1)
    all_rows.append({"config": "equal_weight_buyhold", "split": "full",
                     "cagr": cagr(bh), "max_dd": max_drawdown(bh),
                     "sharpe": ann_sharpe(bh.pct_change().dropna(), 365.25)})

    count = 0
    for W, k, P, use_flow, use_vol in itertools.product(w_vals, k_vals, p_vals, flow_vals, vol_vals):
        sig = sig_cache[W]
        ftag = "f" if use_flow else ""
        vtag = "v" if use_vol else ""
        label = f"W{W}_k{k}_P{P}{'_'+ftag if ftag else ''}{'_'+vtag if vtag else ''}"
        params = {"W": W, "k": k, "T": 90, "P": P,
                  "use_flow": use_flow, "use_vol_filter": use_vol}
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        count += 1
        if count % 50 == 0:
            print(f"  进度: {count}/{total}")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTDIR / "grid_results.csv", index=False)

    full = res[res.split == "full"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")

    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate",
                 "avg_net_bp"]].copy()
    view["train"] = train["cagr"]
    view["valid"] = valid["cagr"]
    view["test"] = test["cagr"]

    print(f"\n\n=== Top-20 全期CAGR ===")
    pd.set_option("display.width", 400)
    top20 = view.sort_values("cagr", ascending=False).head(20)
    print(top20.round(4).to_string())

    print(f"\n\n=== Top-20 test期CAGR ===")
    top20_test = view.sort_values("test", ascending=False).head(20)
    print(top20_test[["cagr", "train", "valid", "test", "max_dd", "sharpe", "n_trades", "win_rate"]].round(4).to_string())

    # 按参数维度统计
    print(f"\n\n=== 参数边际效应 ===")
    for dim, vals in [("W", w_vals), ("k", k_vals), ("P", p_vals)]:
        print(f"\n{dim}:")
        for v in vals:
            subset = [c for c in view.index if f"{dim}{v}" in c or f"{dim}{v}_" in c]
            if subset:
                avg_cagr = view.loc[subset, "cagr"].mean()
                max_cagr = view.loc[subset, "cagr"].max()
                print(f"  {dim}={v}: 均值={avg_cagr*100:.1f}%  最优={max_cagr*100:.1f}%")

    # flow vs no-flow
    flow_cagr = view.loc[[c for c in view.index if "_f" in c], "cagr"].mean()
    noflow_cagr = view.loc[[c for c in view.index if "_f" not in c], "cagr"].mean()
    print(f"\nflow: 均值={flow_cagr*100:.1f}% vs no_flow: {noflow_cagr*100:.1f}%")

    vol_cagr = view.loc[[c for c in view.index if "_v" in c], "cagr"].mean()
    novol_cagr = view.loc[[c for c in view.index if "_v" not in c], "cagr"].mean()
    print(f"vol_filter: 均值={vol_cagr*100:.1f}% vs no_vol: {novol_cagr*100:.1f}%")

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
