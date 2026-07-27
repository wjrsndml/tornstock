"""exp_fine_tune.py — 微调: W=25 + 最优k/P 验证。"""

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_fine")
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
    use_flow = params.get("use_flow", False)
    W_hours = params.get("W_hours", 30 * 24)

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

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
    # 基准
    sig30 = {}
    sig30["z"] = (px - px.rolling(30 * 24).mean()) / px.rolling(30 * 24).std()
    sig30["flow"] = sh.pct_change(5 * 24)

    rows = run_config(px, sig30, {"k": 1.5, "T": 90 * 24, "P": 2, "use_flow": True}, "h_W30_k1.5_P2_f")
    all_rows.extend(rows)
    print("完成 基准 W30")

    # W=25
    for W in [22, 25, 28, 35]:
        W_hours = W * 24
        sig = {}
        sig["z"] = (px - px.rolling(W_hours).mean()) / px.rolling(W_hours).std()
        sig["flow"] = sh.pct_change(5 * 24)

        for k in [1.2, 1.5]:
            for P in [2, 3]:
                label = f"h_W{W}_k{k}_P{P}_f"
                rows = run_config(px, sig, {"k": k, "T": 90 * 24, "P": P, "use_flow": True}, label)
                all_rows.extend(rows)
                print(f"完成 {label}")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTDIR / "results.csv", index=False)

    full = res[res.split == "full"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")

    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate",
                 "avg_net_bp", "avg_hold_d"]].copy()
    view["train"] = train["cagr"]
    view["valid"] = valid["cagr"]
    view["test"] = test["cagr"]

    print("\n=== W微调 + 小时级 ===")
    pd.set_option("display.width", 400)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
