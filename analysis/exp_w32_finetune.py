"""exp_w32_finetune.py — 实验7: W=32附近精搜 + dd_20残差信号叠加 + HMM探索。

三个子实验:
A. W=31/33/34 + k=0.9/1.0/1.1 精细网格 (小时级)
B. dd_20作为入场过滤: z<-k AND dd_20<-threshold
C. z_60趋势作为入场确认 (之前全否, 但W=32时可能不同)

用法: .venv/bin/python analysis/exp_w32_finetune.py
"""

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_w32fine")
INIT_CAPITAL = 1.0


def load_panel_hourly():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1h")
        shares[sym] = df["total_shares"].resample("1h").last().reindex(closes[sym].index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def simulate(px, sig, params):
    dates = px.index
    P = params["P"]
    k = params["k"]
    T = params.get("T", 90 * 24)
    use_flow = params.get("use_flow", True)
    use_dd_filter = params.get("use_dd_filter", False)
    dd_k = params.get("dd_k", -0.05)

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    W_hours = params.get("W_hours", 32 * 24)
    warmup = W_hours + 100
    start_i = warmup

    for i in range(start_i, len(dates)):
        today = px.iloc[i]
        today_z = sig["z"].iloc[i]

        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym, "entry_date": dates[pos[sym]["entry_i"]],
                    "exit_date": dates[i],
                    "net_ret": today[sym] * (1 - SELL_TAX) / pos[sym]["entry_px"] - 1,
                    "hold_days": (i - pos[sym]["entry_i"]) / 24,
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
            held_hours = i - p["entry_i"]
            z_val = today_z[sym]
            exit_now = not np.isnan(z_val) and z_val >= 0
            if exit_now or held_hours >= T:
                pend_sell.append(sym)

        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = today_z.dropna()
            cand = cand[(cand < -k) & ~cand.index.isin(pos)]
            if use_flow:
                fl = sig["flow"].iloc[i]
                cand = cand[fl.reindex(cand.index) > 0]
            if use_dd_filter and "dd_20" in sig:
                dd = sig["dd_20"].iloc[i]
                for sym in list(cand.index):
                    if not pd.isna(dd[sym]) and dd[sym] > dd_k:
                        cand = cand.drop(sym)
            ranked = cand.sort_values()
            pend_buy = list(ranked.index[:slots])

    eq = pd.Series(equity, index=dates[start_i:])
    return eq, trades


def run_config(px, sig, params, label):
    eq, trades = simulate(px, sig, params)
    if len(eq) < 30:
        return []
    eq_d = eq.resample("1D").last().dropna()
    if len(eq_d) < 30:
        return []
    eq_d = eq_d / eq_d.iloc[0]

    rows = []
    spans = {"train": (eq_d.index[0], SPLIT_TRAIN_END),
             "valid": (SPLIT_TRAIN_END, SPLIT_VALID_END),
             "test": (SPLIT_VALID_END, eq_d.index[-1])}
    td = pd.DataFrame(trades)
    for name, (a, b) in spans.items():
        sub = eq_d[(eq_d.index >= a) & (eq_d.index < b)]
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
    full_m = {"config": label, "split": "full", "cagr": cagr(eq_d),
              "max_dd": max_drawdown(eq_d),
              "sharpe": ann_sharpe(eq_d.pct_change().dropna(), 365.25),
              "n_trades": len(td)}
    if len(td):
        full_m.update({"win_rate": (td.net_ret > 0).mean(),
                       "avg_net_bp": td.net_ret.mean() * 1e4,
                       "avg_hold_d": td.hold_days.mean()})
    rows.append(full_m)
    return rows


def main():
    px, sh = load_panel_hourly()

    all_rows = []

    # ── A. W精搜 ──
    for W in [31, 33, 34]:
        W_hours = W * 24
        sig = {}
        sig["z"] = (px - px.rolling(W_hours).mean()) / px.rolling(W_hours).std()
        sig["flow"] = sh.pct_change(5 * 24)
        for k in [0.9, 1.0, 1.1]:
            for P in [2, 3]:
                label = f"h_W{W}_k{k}_P{P}_f"
                params = {"W_hours": W_hours, "k": k, "T": 90 * 24, "P": P, "use_flow": True}
                rows = run_config(px, sig, params, label)
                all_rows.extend(rows)
                print(f"完成A {label}")

    # ── B. dd_20过滤 ──
    sig32 = {}
    sig32["z"] = (px - px.rolling(32 * 24).mean()) / px.rolling(32 * 24).std()
    sig32["flow"] = sh.pct_change(5 * 24)
    sig32["dd_20"] = px / px.rolling(20 * 24).max() - 1

    for k in [1.0, 1.2]:
        for dd_thresh in [-0.01, -0.02, -0.03]:
            for P in [3]:
                label = f"h_W32_k{k}_P{P}_f_dd{abs(dd_thresh):.0%}"
                params = {"W_hours": 32 * 24, "k": k, "T": 90 * 24, "P": P,
                          "use_flow": True, "use_dd_filter": True, "dd_k": dd_thresh}
                rows = run_config(px, sig32, params, label)
                all_rows.extend(rows)
                print(f"完成B {label}")

    # ── 基准 ──
    rows = run_config(px, sig32, {"W_hours": 32 * 24, "k": 1.0, "T": 90 * 24, "P": 3, "use_flow": True}, "h_W32_k1.0_P3_f_BASE")
    all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTDIR / "results.csv", index=False)

    full = res[res.split == "full"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")

    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate",
                 "avg_net_bp"]].copy()
    view["train"] = train["cagr"]
    view["valid"] = valid["cagr"]
    view["test"] = test["cagr"]

    print("\n=== W精搜 + dd_20过滤 ===")
    pd.set_option("display.width", 400)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    base_cagr = full.loc["h_W32_k1.0_P3_f_BASE", "cagr"]
    print(f"\n基准 W32_k1.0_P3_f: {base_cagr*100:.1f}%")
    for cfg in view.sort_values("cagr", ascending=False).index:
        if cfg == "h_W32_k1.0_P3_f_BASE":
            continue
        d = view.loc[cfg, "cagr"] - base_cagr
        if abs(d) > 0.002:
            marker = "★" if d > 0.005 else ("✓" if d > 0 else "✗")
            print(f"  {marker} {cfg}: {view.loc[cfg, 'cagr']*100:.1f}% (Δ={d*100:+.1f}pp)")

    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
