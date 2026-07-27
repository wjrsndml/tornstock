"""exp_mtf_resonance.py — 实验2: 多时间框架z-score共振 + 波动率状态过滤。

核心假设:
1. 多时间框架(10/30/60日)z-score共振的入场信号质量高于单窗口
2. 高波动率时回复路径更长、假突破更多,低波动率时的深z才是干净信号

实验设计:
- 共振定义: z_10, z_30, z_60 同时<0为弱信号, 同时<-1为强信号
- 波动率分位数: 当前20日波动率 / 过去200日波动率中位数, >1.2倍时收紧仓位
- z_60 趋势拐头: z_60不再创新低作为入场确认信号
- 对比变体: 纯共振 / 共振+波动过滤 / 共振+趋势确认 / 全部叠加

用法: .venv/bin/python analysis/exp_mtf_resonance.py
"""

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_mtf")
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


def build_signals(px: pd.DataFrame, sh: pd.DataFrame) -> dict:
    """多时间框架z-score + 波动率状态"""
    sig = {}
    # 多窗口z-score
    for w in [10, 30, 60]:
        mean = px.rolling(w, min_periods=5).mean()
        std = px.rolling(w, min_periods=5).std().replace(0, np.nan)
        sig[f"z_{w}"] = (px - mean) / std

    # 波动率状态: 当前vol / 长期vol中位数
    vol_20 = px.pct_change().rolling(20).std() * np.sqrt(365.25)
    vol_200_median = px.pct_change().rolling(200).std().expanding().median() * np.sqrt(365.25)
    sig["vol_ratio"] = vol_20 / vol_200_median.replace(0, np.nan)

    # z_60变化 (用于趋势确认)
    sig["z60_delta"] = sig["z_60"] - sig["z_60"].shift(3)  # 3天变化

    # 资金流
    sig["flow"] = sh.pct_change(5)

    return sig


def check_resonance(sig, col, level="weak"):
    """检查多时间框架共振。level='weak': 三个z都<0; level='strong': 三个z都<-1"""
    z10 = sig["z_10"][col]
    z30 = sig["z_30"][col]
    z60 = sig["z_60"][col]
    if pd.isna(z10) or pd.isna(z30) or pd.isna(z60):
        return False
    if level == "weak":
        return z10 < 0 and z30 < 0 and z60 < 0
    else:
        return z10 < -1 and z30 < -1 and z60 < -1


def simulate(px, sig, strategy, params):
    """事件驱动模拟，支持多种过滤条件。"""
    dates = px.index
    P = params["P"]
    k = params.get("k", 1.5)
    T = params.get("T", 90)
    use_flow = params.get("use_flow", False)
    use_vol_filter = params.get("use_vol_filter", False)
    use_trend_confirm = params.get("use_trend_confirm", False)
    entry_level = params.get("entry_level", "z30")  # z30 / resonance_weak / resonance_strong

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    warmup = 100
    start_i = warmup

    for i in range(start_i, len(dates)):
        today = px.iloc[i]

        # T+1成交
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

        # 净值
        eq = cash + sum(p["shares"] * today[s] for s, p in pos.items())
        equity.append(eq)

        # 退出规则
        held = list(pos.keys())
        for sym in held:
            p = pos[sym]
            held_days = i - p["entry_i"]
            z30 = sig["z_30"].iloc[i][sym]
            exit_now = not np.isnan(z30) and z30 >= 0
            if exit_now or held_days >= T:
                pend_sell.append(sym)

        # 入场规则
        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand_scores = {}  # sym -> score (越小越好)
            for sym in px.columns:
                if sym in pos:
                    continue
                z30 = sig["z_30"].iloc[i][sym]
                if pd.isna(z30):
                    continue

                eligible = False
                score = z30  # 默认按z_30排序

                if entry_level == "z30":
                    eligible = z30 < -k
                elif entry_level == "resonance_weak":
                    eligible = check_resonance(sig, sym, "weak") and z30 < -k
                elif entry_level == "resonance_strong":
                    eligible = check_resonance(sig, sym, "strong")

                if not eligible:
                    continue

                # 波动率过滤：vol过高时要求更深的z
                if use_vol_filter:
                    vol_r = sig["vol_ratio"].iloc[i][sym]
                    if not pd.isna(vol_r) and vol_r > 1.2:
                        if z30 > -(k + 0.5):  # 波动高的时期提高入场门槛
                            continue

                # 趋势确认：z_60不再创新低
                if use_trend_confirm:
                    z60_d = sig["z60_delta"].iloc[i][sym]
                    if not pd.isna(z60_d) and z60_d <= 0:
                        continue  # z_60还在下降，跳过

                # 资金流过滤
                if use_flow:
                    fl = sig["flow"].iloc[i][sym]
                    if pd.isna(fl) or fl <= 0:
                        continue

                cand_scores[sym] = score

            ranked = sorted(cand_scores.items(), key=lambda x: x[1])
            pend_buy = [s for s, _ in ranked[:slots]]

    eq = pd.Series(equity, index=dates[start_i:])
    return eq, trades


def run_config(px, sig, strategy, params, label):
    eq, trades = simulate(px, sig, strategy, params)
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
    sig = build_signals(px, sh)

    all_rows = []
    # 基准
    bh = (px / px.iloc[0]).mean(axis=1)
    all_rows.append({"config": "equal_weight_buyhold", "split": "full",
                     "cagr": cagr(bh), "max_dd": max_drawdown(bh),
                     "sharpe": ann_sharpe(bh.pct_change().dropna(), 365.25)})

    # 基准配置 (原始z30策略)
    base_configs = [
        ("baseline_z30_k1.5_P3", "z30", {"k": 1.5, "T": 90, "P": 3, "use_flow": False}),
        ("baseline_z30_k1.0_P5", "z30", {"k": 1.0, "T": 90, "P": 5, "use_flow": False}),
        ("baseline_zflow_k1.5_P3", "z30", {"k": 1.5, "T": 90, "P": 3, "use_flow": True}),
    ]
    for label, strategy, params in base_configs:
        rows = run_config(px, sig, strategy, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # 实验配置
    exp_configs = [
        # 纯共振（弱版: 三窗口同时<0 + z30<-k）
        ("resonance_weak_k1.5_P3", "resonance_weak",
         {"k": 1.5, "T": 90, "P": 3}),
        ("resonance_weak_k1.0_P3", "resonance_weak",
         {"k": 1.0, "T": 90, "P": 3}),
        # 强共振 (三窗口同时<-1)
        ("resonance_strong_P3", "resonance_strong",
         {"k": 1.0, "T": 90, "P": 3}),
        ("resonance_strong_P5", "resonance_strong",
         {"k": 1.0, "T": 90, "P": 5}),
        # 共振 + 波动过滤
        ("res_weak_vol_k1.5_P3", "resonance_weak",
         {"k": 1.5, "T": 90, "P": 3, "use_vol_filter": True}),
        ("res_weak_vol_k1.0_P5", "resonance_weak",
         {"k": 1.0, "T": 90, "P": 5, "use_vol_filter": True}),
        # 共振 + 趋势确认
        ("res_weak_trend_k1.5_P3", "resonance_weak",
         {"k": 1.5, "T": 90, "P": 3, "use_trend_confirm": True}),
        ("res_weak_trend_k1.0_P5", "resonance_weak",
         {"k": 1.0, "T": 90, "P": 5, "use_trend_confirm": True}),
        # 全部叠加
        ("res_full_k1.5_P3", "resonance_weak",
         {"k": 1.5, "T": 90, "P": 3, "use_vol_filter": True,
          "use_trend_confirm": True}),
        # z30 + 波动过滤 (不要求共振)
        ("z30_vol_k1.5_P3", "z30",
         {"k": 1.5, "T": 90, "P": 3, "use_vol_filter": True}),
    ]

    for label, strategy, params in exp_configs:
        rows = run_config(px, sig, strategy, params, label)
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

    print("\n\n=== 多时间框架共振实验结果 ===")
    pd.set_option("display.width", 300)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    # 对比增益
    print("\n\n=== 相对基准的CAGR增益 (full期) ===")
    for base_label, _, _ in base_configs:
        if base_label in full.index:
            base_cagr = full.loc[base_label, "cagr"]
            for exp_label, _, _ in exp_configs:
                if exp_label in full.index:
                    delta = full.loc[exp_label, "cagr"] - base_cagr
                    if abs(delta) > 0.001:
                        marker = "★" if delta > 0.01 else ("✓" if delta > 0 else "✗")
                        print(f"  {marker} {exp_label} vs {base_label}: Δcagr={delta:+.4f} ({delta*100:+.1f}pp)")

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
