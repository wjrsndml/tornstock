"""exp_ou_adaptive.py — 实验1: OU过程参数校准 + 自适应z-score窗口 + 轮动回测。

核心假设: 不同股票的均值回复速度(θ)不同, 使用固定的30日窗口对所有股票
计算z-score是次优的。本实验对每只股票单独拟合OU过程参数, 用自适应窗口
替代固定窗口, 检验能否提升轮动策略的收益率。

方法:
1. AR(1)回归: X_{t+1} = a + b*X_t + ε_t
2. 从b反推θ: b = e^{-θΔt} → θ = -ln(b)/Δt
3. 回复半衰期 = ln(2)/θ
4. 自适应z-score窗口 = clip(half_life * 2, 10, 60)天
5. 对自适应窗口和固定30日窗口分别回测对比

用法: .venv/bin/python analysis/exp_ou_adaptive.py
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = Path(__file__).resolve().parent / "output" / "exp_ou"
INIT_CAPITAL = 1.0

# ── OU 参数估计 ──


def fit_ou_ar1(price_series: pd.Series) -> dict:
    """用AR(1)回归拟合OU过程参数。

    离散形式: X_{t+1} = a + b*X_t + ε_t
    OU参数: θ = -ln(b)/Δt, μ = a/(1-b), σ = std(ε)*sqrt(2θ/(1-b²))
    半衰期(天) = ln(2)/θ
    """
    x = price_series.values
    x_t = x[:-1]
    x_t1 = x[1:]
    # AR(1): x_t1 = a + b * x_t + eps
    X = np.column_stack([np.ones_like(x_t), x_t])
    coeff = np.linalg.lstsq(X, x_t1, rcond=None)[0]
    a, b = coeff[0], coeff[1]
    residuals = x_t1 - (a + b * x_t)
    sigma_eps = np.std(residuals)

    theta = -np.log(max(b, 0.001))  # Δt=1天
    mu = a / (1 - b) if abs(1 - b) > 1e-8 else np.mean(x)
    half_life = np.log(2) / max(theta, 1e-6)

    return {
        "theta": theta, "mu": mu, "b": b, "a": a,
        "sigma_eps": sigma_eps,
        "half_life_days": half_life,
        "adaptive_window": int(np.clip(half_life * 2, 10, 60)),
    }


def compute_zscore(px: pd.Series, window: int) -> pd.Series:
    """滚动z-score: (price - rolling_mean) / rolling_std"""
    roll_mean = px.rolling(window, min_periods=5).mean()
    roll_std = px.rolling(window, min_periods=5).std()
    return (px - roll_mean) / roll_std.replace(0, np.nan)


# ── 数据加载 ──


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """日线面板 + 每只股票的OU拟合结果。"""
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    ou_params = {}
    for sym in stocks:
        df = load_stock(sym)
        daily = resample_close(df, "1D")
        closes[sym] = daily
        shares[sym] = df["total_shares"].resample("1D").last().reindex(daily.index)
        # OU拟合: 只在train期内做, 避免未来函数
        train_daily = daily[daily.index < SPLIT_TRAIN_END]
        ou_params[sym] = fit_ou_ar1(train_daily)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill(), ou_params


def build_signals(px: pd.DataFrame, sh: pd.DataFrame,
                  ou_params: dict, adaptive: bool) -> dict[str, pd.DataFrame]:
    """计算信号矩阵。adaptive=True时使用股票特定窗口。"""
    if adaptive:
        z_df = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
        window_map = {}
        for sym in px.columns:
            w = ou_params[sym]["adaptive_window"]
            window_map[sym] = w
            z_df[sym] = compute_zscore(px[sym], w)
    else:
        W = 30
        z_df = (px - px.rolling(W).mean()) / px.rolling(W).std()
        window_map = {}

    return {
        "dd": px / px.rolling(30).max() - 1,
        "z": z_df,
        "flow": sh.pct_change(5),
        "window_map": window_map,
    }


# ── 回测引擎(与06_rotation相同的逻辑,增加自适应窗口变体) ──


def simulate(px: pd.DataFrame, sig: dict[str, pd.DataFrame],
             strategy: str, params: dict) -> tuple[pd.Series, list[dict]]:
    """事件驱动模拟。"""
    dates = px.index
    cols = px.columns
    P = params["P"]
    cash = INIT_CAPITAL
    pos: dict[str, dict] = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    # 确定预热期 (z-score窗口可能因自适应而不同)
    z = sig["z"]
    first_valid = z.notna().any(axis=1)
    first_valid_idx = first_valid[first_valid].index[0]
    start_i = max(px.index.get_loc(first_valid_idx) + 10, 60)

    for i in range(start_i, len(dates)):
        today = px.iloc[i]

        # T+1成交
        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym,
                    "entry_date": dates[pos[sym]["entry_i"]],
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

        # 决策
        held = list(pos.keys())
        for sym in held:
            p = pos[sym]
            held_days = i - p["entry_i"]
            z_val = sig["z"].iloc[i][sym]
            exit_now = not np.isnan(z_val) and z_val >= 0
            if exit_now or held_days >= params["T"]:
                pend_sell.append(sym)

        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = sig["z"].iloc[i].dropna()
            cand = cand[(cand < -params["k"]) & ~cand.index.isin(pos)]
            if strategy == "zflow":
                fl = sig["flow"].iloc[i]
                cand = cand[fl.reindex(cand.index) > 0]
            ranked = cand.sort_values()
            pend_buy = list(ranked.index[:slots])

    eq = pd.Series(equity, index=dates[start_i:])
    return eq, trades


def run_config(px, sig, strategy, params, label):
    """运行单个配置, 返回分段的metrics记录。"""
    from common import SPLIT_TRAIN_END, SPLIT_VALID_END

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
        m = {
            "config": label, "split": name,
            "cagr": cagr(sub), "max_dd": max_drawdown(sub),
            "sharpe": ann_sharpe(sub.pct_change().dropna(), 365.25),
            "n_trades": len(t),
        }
        if len(t):
            m.update({
                "win_rate": (t.net_ret > 0).mean(),
                "avg_net_bp": t.net_ret.mean() * 1e4,
                "avg_hold_d": t.hold_days.mean(),
            })
        rows.append(m)

    # full
    full_eq = eq / eq.iloc[0]
    full_m = {
        "config": label, "split": "full",
        "cagr": cagr(full_eq), "max_dd": max_drawdown(full_eq),
        "sharpe": ann_sharpe(full_eq.pct_change().dropna(), 365.25),
        "n_trades": len(td),
    }
    if len(td):
        full_m.update({
            "win_rate": (td.net_ret > 0).mean(),
            "avg_net_bp": td.net_ret.mean() * 1e4,
            "avg_hold_d": td.hold_days.mean(),
        })
    rows.append(full_m)
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    px, sh, ou_params = load_panel()

    # ── 输出OU拟合结果 ──
    ou_df = pd.DataFrame(ou_params).T
    ou_df.index.name = "stock"
    ou_df.to_csv(OUTDIR / "ou_params.csv")
    print("=== OU参数拟合结果(按半衰期排序) ===")
    print(ou_df.sort_values("half_life_days")[["theta", "mu", "half_life_days",
          "adaptive_window", "sigma_eps"]].round(4).to_string())
    print(f"\n半衰期范围: {ou_df.half_life_days.min():.0f} - {ou_df.half_life_days.max():.0f} 天")
    print(f"自适应窗口范围: {ou_df.adaptive_window.min():.0f} - {ou_df.adaptive_window.max():.0f} 天")
    print(f"固定窗口: 30 天")

    # ── 生成两套信号 ──
    sig_adaptive = build_signals(px, sh, ou_params, adaptive=True)
    sig_fixed = build_signals(px, sh, ou_params, adaptive=False)

    # ── 回测对比 ──
    all_rows = []

    # 基准:等权买入持有
    bh = (px / px.iloc[0]).mean(axis=1)
    all_rows.append({"config": "equal_weight_buyhold", "split": "full",
                     "cagr": cagr(bh), "max_dd": max_drawdown(bh),
                     "sharpe": ann_sharpe(bh.pct_change().dropna(), 365.25)})

    configs = [
        ("zscore", {"k": 1.0, "T": 90, "P": 3}),
        ("zscore", {"k": 1.5, "T": 90, "P": 3}),
        ("zscore", {"k": 1.0, "T": 90, "P": 5}),
        ("zscore", {"k": 1.5, "T": 90, "P": 5}),
        ("zflow", {"k": 1.5, "T": 90, "P": 3}),
        ("zflow", {"k": 1.0, "T": 90, "P": 3}),
    ]

    for strategy, params in configs:
        # 自适应窗口
        label_ada = f"ada_{strategy}_" + "_".join(f"{k}{v}" for k, v in params.items())
        rows = run_config(px, sig_adaptive, strategy, params, label_ada)
        all_rows.extend(rows)
        print(f"完成 {label_ada}")

        # 固定30日窗口
        label_fixed = f"fix_{strategy}_" + "_".join(f"{k}{v}" for k, v in params.items())
        rows = run_config(px, sig_fixed, strategy, params, label_fixed)
        all_rows.extend(rows)
        print(f"完成 {label_fixed}")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUTDIR / "results.csv", index=False)

    # ── 比较打印 ──
    full = res[res.split == "full"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")

    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate",
                 "avg_net_bp"]].copy()
    view["train"] = train["cagr"]
    view["valid"] = valid["cagr"]
    view["test"] = test["cagr"]

    print("\n\n=== 自适应窗口 vs 固定30日窗口 (全期CAGR排序) ===")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 20)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    # ── 对比增益 ──
    print("\n\n=== 自适应窗口增益 (ada vs fix, full期) ===")
    for strategy, params in configs:
        label_base = "_".join(f"{k}{v}" for k, v in params.items())
        ada_key = f"ada_{strategy}_{label_base}"
        fix_key = f"fix_{strategy}_{label_base}"
        if ada_key in full.index and fix_key in full.index:
            delta = full.loc[ada_key, "cagr"] - full.loc[fix_key, "cagr"]
            delta_test = test.loc[ada_key, "cagr"] - test.loc[fix_key, "cagr"]
            print(f"  {strategy}_{label_base}: Δfull={delta:+.4f} ({delta*100:+.1f}pp) | Δtest={delta_test:+.4f} ({delta_test*100:+.1f}pp)")

    # ── 最优配置详情 ──
    best = view.sort_values("cagr", ascending=False).iloc[0]
    print(f"\n=== 最优配置: {best.name} ===")
    print(f"  全期CAGR: {best.cagr*100:.1f}%  | test: {best.test*100:.1f}%")
    print(f"  maxDD: {best.max_dd*100:.2f}%  | Sharpe: {best.sharpe:.2f}")
    print(f"  胜率: {best.win_rate*100:.1f}%  | 笔数: {best.n_trades}")
    print(f"  平均每笔: {best.avg_net_bp:.1f}bp")

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
