"""exp_exit_optimize.py — 实验4+5: 退出规则优化 + 仓位管理 + 综合改进。

核心假设: 当前z≥0/90天超时的固定退出规则是主要可改进点。
相比入场规则(已经被z-score的强IC保护), 退出规则更可能提供增量。

实验维度:
A. 退出规则变体:
  1. trailing_z: z回复Δz就退出 (如z_entry=-2.0, Δz=1.0 → z≥-1.0退出)
  2. time_decay: 持有时间越长, 退出阈值越宽松
  3. vol_scaled: 高波动率时提前退出
  4. competition: 有更好机会时换仓
  5. profit_target: 达到固定止盈就退出
  6. combined: 以上组合

B. 仓位管理:
  1. 信号强度仓位: z越深, 分配越多资金
  2. Kelly近似: 基于历史和当前波动率调整仓位

C. 综合改进: 波动率过滤 + 最优退出 + 信号强度仓位

用法: .venv/bin/python analysis/exp_exit_optimize.py
"""

import itertools
import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_exit")
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


def build_signals(px, sh):
    sig = {}
    W = 30
    sig["z"] = (px - px.rolling(W).mean()) / px.rolling(W).std()
    sig["flow"] = sh.pct_change(5)
    sig["vol_20"] = px.pct_change().rolling(20).std()
    vol_200_med = px.pct_change().rolling(200).std().expanding().median()
    sig["vol_ratio"] = sig["vol_20"] / vol_200_med.replace(0, np.nan)
    sig["dd_20"] = px / px.rolling(20).max() - 1
    return sig


def simulate(px, sig, params):
    """可配置退出规则和仓位管理的模拟器。"""
    dates = px.index
    P = params["P"]
    k = params.get("k", 1.5)
    T = params.get("T", 90)
    use_flow = params.get("use_flow", False)
    use_vol_filter = params.get("use_vol_filter", False)
    exit_rule = params.get("exit_rule", "z_zero")  # z_zero / trailing / time_decay / vol_scaled / competition / profit_target / combined
    trailing_dz = params.get("trailing_dz", 1.0)
    profit_target = params.get("profit_target", 0.005)
    position_sizing = params.get("position_sizing", "equal")  # equal / signal_strength

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
        today_vol = sig["vol_ratio"].iloc[i]

        # T+1成交 - 卖出
        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym, "entry_date": dates[pos[sym]["entry_i"]],
                    "exit_date": dates[i],
                    "net_ret": today[sym] * (1 - SELL_TAX) / pos[sym]["entry_px"] - 1,
                    "hold_days": i - pos[sym]["entry_i"],
                    "exit_reason": pos[sym].get("exit_reason", ""),
                })
                del pos[sym]

        # T+1成交 - 买入
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                if position_sizing == "equal":
                    alloc = cash / (n_slots - j)
                elif position_sizing == "signal_strength":
                    z_val = today_z[sym]
                    if not pd.isna(z_val):
                        weight = min(abs(z_val) / k, 2.0)  # z越深仓位越大, 上限2x
                    else:
                        weight = 1.0
                    alloc = cash * weight / (P * 1.2)  # 归一化
                    alloc = min(alloc, cash * 0.6)  # 上限60%资金
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i, "entry_z": today_z[sym]}
                cash -= alloc
        pend_sell, pend_buy = [], []

        # 净值
        eq = cash + sum(p["shares"] * today[s] for s, p in pos.items())
        equity.append(eq)

        # 退出决策
        held = list(pos.keys())
        for sym in held:
            p = pos[sym]
            held_days = i - p["entry_i"]
            current_z = today_z[sym]
            current_pnl = today[sym] / p["entry_px"] - 1
            exit_now = False
            exit_reason = ""

            if pd.isna(current_z):
                exit_now = held_days >= T

            elif exit_rule == "z_zero":
                exit_now = current_z >= 0

            elif exit_rule == "trailing":
                entry_z = p.get("entry_z", -k)
                exit_now = current_z >= entry_z + trailing_dz

            elif exit_rule == "time_decay":
                # 持有时间越长, 阈值从0降到-0.5
                threshold = -0.5 * min(held_days / T, 1.0)
                exit_now = current_z >= threshold

            elif exit_rule == "vol_scaled":
                vr = today_vol[sym] if not pd.isna(today_vol[sym]) else 1.0
                threshold = 0.3 * (vr - 1.0)  # 高波动率时正阈值(早出), 低波动率时阴性阈值(晚出)
                exit_now = current_z >= threshold

            elif exit_rule == "competition":
                # 如果有更好机会(其他股票z低1.5以上)且持有>7天, 换仓
                other_z = today_z.drop(sym).dropna()
                best_other = other_z.min()
                if held_days > 7 and not pd.isna(best_other) and best_other < current_z - 1.5:
                    exit_now = True
                    exit_reason = "compete"
                elif current_z >= 0:
                    exit_now = True

            elif exit_rule == "profit_target":
                exit_now = current_pnl >= profit_target or current_z >= 0

            elif exit_rule == "combined":
                # 止盈优先
                if current_pnl >= profit_target:
                    exit_now = True
                    exit_reason = "profit"
                # 竞争换仓
                elif held_days > 7:
                    other_z = today_z.drop(sym).dropna()
                    best_other = other_z.min()
                    if not pd.isna(best_other) and best_other < current_z - 1.5:
                        exit_now = True
                        exit_reason = "compete"
                # 默认z_zero
                elif current_z >= 0:
                    exit_now = True
                    exit_reason = "z_zero"

            if exit_now or held_days >= T:
                p["exit_reason"] = exit_reason if exit_reason else ("timeout" if held_days >= T else exit_rule)
                pend_sell.append(sym)

        # 入场决策
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
            if "exit_reason" in t.columns:
                m["timeout_pct"] = (t.exit_reason == "timeout").mean()
                m["compete_pct"] = (t.exit_reason == "compete").mean()
                m["profit_exit_pct"] = (t.exit_reason == "profit").mean()
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
    sig = build_signals(px, sh)

    all_rows = []
    # 基准
    bh = (px / px.iloc[0]).mean(axis=1)
    all_rows.append({"config": "equal_weight_buyhold", "split": "full",
                     "cagr": cagr(bh), "max_dd": max_drawdown(bh),
                     "sharpe": ann_sharpe(bh.pct_change().dropna(), 365.25)})

    # ── 基准策略 ──
    for label, params in [
        ("baseline_z_k1.5_P3", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "z_zero"}),
        ("baseline_zflow_k1.5_P3", {"k": 1.5, "T": 90, "P": 3, "use_flow": True, "exit_rule": "z_zero"}),
        ("baseline_z_k1.0_P5", {"k": 1.0, "T": 90, "P": 5, "exit_rule": "z_zero"}),
    ]:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成基线 {label}")

    # ── 退出规则实验(A) ──
    exit_experiments = [
        # trailing Δz
        ("trail_dz1.0", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing", "trailing_dz": 1.0}),
        ("trail_dz1.5", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing", "trailing_dz": 1.5}),
        ("trail_dz0.5", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing", "trailing_dz": 0.5}),
        # time_decay
        ("tdecay", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "time_decay"}),
        # vol_scaled
        ("vol_exit", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "vol_scaled"}),
        # competition
        ("compete", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "competition"}),
        # profit_target
        ("profit_0.5pct", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "profit_target", "profit_target": 0.005}),
        ("profit_0.8pct", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "profit_target", "profit_target": 0.008}),
        # combined
        ("combined_0.5pct", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "combined", "profit_target": 0.005}),
        ("combined_0.8pct", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "combined", "profit_target": 0.008}),
    ]
    for label, params in exit_experiments:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── 退出规则 + 波动率过滤 + 资金流(B) ──
    enhanced = [
        ("trail_dz1.0_vol", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing",
         "trailing_dz": 1.0, "use_vol_filter": True}),
        ("trail_dz1.0_vol_flow", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing",
         "trailing_dz": 1.0, "use_vol_filter": True, "use_flow": True}),
        ("combined_vol_flow", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "combined",
         "profit_target": 0.008, "use_vol_filter": True, "use_flow": True}),
        ("compete_vol_flow", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "competition",
         "use_vol_filter": True, "use_flow": True}),
    ]
    for label, params in enhanced:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── 信号强度仓位(C) ──
    sizing = [
        ("z_zero_sigsize", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "z_zero",
         "position_sizing": "signal_strength"}),
        ("trail_dz1.0_sigsize", {"k": 1.5, "T": 90, "P": 3, "exit_rule": "trailing",
         "trailing_dz": 1.0, "position_sizing": "signal_strength", "use_vol_filter": True}),
    ]
    for label, params in sizing:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── 大P + 综合(C) ──
    large_p = [
        ("combined_vol_flow_P5", {"k": 1.5, "T": 90, "P": 5, "exit_rule": "combined",
         "profit_target": 0.008, "use_vol_filter": True, "use_flow": True}),
        ("trail_dz1.0_vol_flow_P5", {"k": 1.5, "T": 90, "P": 5, "exit_rule": "trailing",
         "trailing_dz": 1.0, "use_vol_filter": True, "use_flow": True}),
    ]
    for label, params in large_p:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── 结果汇总 ──
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

    print("\n\n=== 退出规则 + 仓位管理优化 ===")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 20)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    # 增益分析
    base_cagr = full.loc["baseline_z_k1.5_P3", "cagr"]
    base_zflow = full.loc["baseline_zflow_k1.5_P3", "cagr"]
    print(f"\n\n=== 相对基准的CAGR增益 ===")
    print(f"baseline_z_k1.5_P3: {base_cagr*100:.1f}%")
    print(f"baseline_zflow_k1.5_P3: {base_zflow*100:.1f}%")
    for cfg in view.sort_values("cagr", ascending=False).index:
        if cfg.startswith("baseline") or cfg == "equal_weight_buyhold":
            continue
        d1 = view.loc[cfg, "cagr"] - base_cagr
        d2 = view.loc[cfg, "cagr"] - base_zflow
        d_test1 = view.loc[cfg, "test"] - view.loc["baseline_z_k1.5_P3", "test"]
        marker = "★" if d2 > 0.01 else ("✓" if d2 > 0 else "✗")
        print(f"  {marker} {cfg:40s}: Δvs_zscore={d1*100:+.1f}pp  Δvs_zflow={d2*100:+.1f}pp  Δtest={d_test1*100:+.1f}pp")

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
