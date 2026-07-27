"""exp_spread_strategy.py — 将协整价差信号叠加到z-score轮动策略中。

在现有最优策略(W=32, k=1.0, P=3, flow)基础上, 加入价差z-score信号:
- spread_z > 0: 股票相对同行"偏贵"(相对动量向上), 降低买入优先级
- spread_z < 0: 股票相对同行"偏便宜", 提高买入优先级

排序键: z_30 + λ × spread_z (λ为混合权重)

用法: .venv/bin/python analysis/exp_spread_strategy.py
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_spread")
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


def build_spread_z(px, top_n_pairs=50):
    """构建综合价差z-score(日频)。"""
    stocks = list(px.columns)
    px_log = np.log(px)

    # 在train期找最显著的协整对
    train_mask = px.index < SPLIT_TRAIN_END
    px_train = px[train_mask]
    px_log_train = np.log(px_train)

    # 对每对做协整检验
    pair_adf = []
    for i, j in itertools.combinations(range(len(stocks)), 2):
        y, x = px_log_train[stocks[i]], px_log_train[stocks[j]]
        valid = y.notna() & x.notna()
        if valid.sum() < 100:
            continue
        X = np.column_stack([np.ones_like(x[valid]), x[valid]])
        coeff = np.linalg.lstsq(X, y[valid], rcond=None)[0]
        alpha, beta = coeff[0], coeff[1]
        spread = y[valid] - (alpha + beta * x[valid])
        # ADF
        from statsmodels.tsa.stattools import adfuller
        try:
            adf = adfuller(spread, maxlag=5, autolag="AIC")[0]
        except:
            continue
        pair_adf.append((adf, stocks[i], stocks[j], beta))

    # 选ADF最显著的top-N对
    pair_adf.sort(key=lambda x: x[0])
    top_pairs = pair_adf[:top_n_pairs]

    # 构建综合价差z
    composite = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    counts = pd.Series(0, index=px.columns)
    for adf_stat, sym_y, sym_x, beta in top_pairs:
        spread_full = px_log[sym_y] - beta * px_log[sym_x]
        spread_ma = spread_full.rolling(30).mean()
        spread_std = spread_full.rolling(30).std().replace(0, np.nan)
        spread_z = (spread_full - spread_ma) / spread_std
        composite[sym_y] = composite[sym_y].fillna(0) - spread_z
        composite[sym_x] = composite[sym_x].fillna(0) + spread_z
        counts[sym_y] += 1
        counts[sym_x] += 1
    for sym in px.columns:
        if counts[sym] > 0:
            composite[sym] /= counts[sym]

    return composite


def simulate(px, sig, params):
    dates = px.index
    P = params["P"]
    k = params["k"]
    T = params.get("T", 90)
    use_flow = params.get("use_flow", True)
    spread_weight = params.get("spread_weight", 0.0)

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    warmup = 100
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
            cand = cand[cand < -k]
            cand = cand[~cand.index.isin(pos)]
            if use_flow:
                fl = sig["flow"].iloc[i]
                cand = cand[fl.reindex(cand.index) > 0]

            # 混合排序: z + λ × spread_z (spread_z为正=偏贵,应降低优先级)
            if spread_weight != 0 and "spread_z" in sig:
                sz = sig["spread_z"].iloc[i]
                # 混合得分: z (越低越好) + λ*spread_z (residual IC>0, 即spread_z高→未来涨→应该买→降低排序键)
                # 实际: 我们想要买z低+spread_z高的股票
                combined = {}
                for sym in cand.index:
                    z_val = cand[sym]
                    s_val = sz.get(sym, 0)
                    if not pd.isna(s_val):
                        combined[sym] = z_val - spread_weight * s_val  # spread高→分数更低→优先买入
                    else:
                        combined[sym] = z_val
                ranked = sorted(combined.items(), key=lambda x: x[1])
            else:
                ranked = cand.sort_values().items() if hasattr(cand, 'items') else [(s, cand[s]) for s in cand.index]
                ranked = sorted(ranked, key=lambda x: x[1])

            pend_buy = [s for s, _ in ranked[:slots]]

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
    full_m = {"config": label, "split": "full", "cagr": cagr(eq / eq.iloc[0]),
              "max_dd": max_drawdown(eq / eq.iloc[0]),
              "sharpe": ann_sharpe((eq / eq.iloc[0]).pct_change().dropna(), 365.25),
              "n_trades": len(td)}
    if len(td):
        full_m.update({"win_rate": (td.net_ret > 0).mean(),
                       "avg_net_bp": td.net_ret.mean() * 1e4,
                       "avg_hold_d": td.hold_days.mean()})
    rows.append(full_m)
    return rows


def main():
    px, sh = load_panel()

    # 价差z信号
    print("构建价差z-score...")
    spread_z = build_spread_z(px, top_n_pairs=50)

    # 单股z信号
    W = 32
    sig = {}
    sig["z"] = (px - px.rolling(W).mean()) / px.rolling(W).std()
    sig["flow"] = sh.pct_change(5)
    sig["spread_z"] = spread_z

    all_rows = []

    # 基准
    for label, params in [
        ("BASE_W32_k1.0_P3_f", {"k": 1.0, "T": 90, "P": 3, "use_flow": True, "spread_weight": 0.0}),
        ("BASE_W32_k1.2_P2_f", {"k": 1.2, "T": 90, "P": 2, "use_flow": True, "spread_weight": 0.0}),
    ]:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # 价差叠加实验 (日线)
    for k in [1.0, 1.2]:
        for P in [2, 3]:
            for w in [0.2, 0.5, 1.0]:
                label = f"spread_w{w}_k{k}_P{P}_f"
                params = {"k": k, "T": 90, "P": P, "use_flow": True, "spread_weight": w}
                rows = run_config(px, sig, params, label)
                all_rows.extend(rows)
                print(f"完成 {label}")

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

    print("\n=== 价差叠加策略 (日线) ===")
    pd.set_option("display.width", 400)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    base_cagr = full.loc["BASE_W32_k1.0_P3_f", "cagr"]
    base2_cagr = full.loc["BASE_W32_k1.2_P2_f", "cagr"]
    print(f"\n基准 k1.0_P3: {base_cagr*100:.1f}%  |  k1.2_P2: {base2_cagr*100:.1f}%")
    for cfg in view.sort_values("cagr", ascending=False).index:
        if "BASE" in cfg:
            continue
        d1 = view.loc[cfg, "cagr"] - base_cagr
        d2 = view.loc[cfg, "cagr"] - base2_cagr
        if abs(d1) > 0.002 or abs(d2) > 0.002:
            marker = "★" if (d1 > 0.005 or d2 > 0.005) else ("✓" if (d1 > 0 or d2 > 0) else "✗")
            print(f"  {marker} {cfg}: {view.loc[cfg, 'cagr']*100:.1f}% (Δk1P3={d1*100:+.1f}pp Δk1.2P2={d2*100:+.1f}pp)")

    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
