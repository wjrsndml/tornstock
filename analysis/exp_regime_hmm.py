"""exp_regime_hmm.py — 实验8: 市场状态识别 + 动态策略切换。

方法:
A. 波动率分位法: 当前vol vs 历史vol, 高分位→更保守(高k), 低分位→更激进(低k)
B. 2状态Gaussian HMM: 用EM算法拟合z-score回复模式, 识别强/弱回复期
C. 性能反馈: 滚动窗口内胜率高→激进, 胜率低→保守

用法: .venv/bin/python analysis/exp_regime_hmm.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_regime")
INIT_CAPITAL = 1.0


def load_panel_hourly():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1h")
        shares[sym] = df["total_shares"].resample("1h").last().reindex(closes[sym].index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def simple_hmm_2state(observations, n_iter=30):
    """手动实现2状态Gaussian HMM(Baum-Welch/EM)。

    observations: 1D array of observed values
    返回: 每个时间点的状态概率 (n_timesteps, 2)
    """
    obs = observations[~np.isnan(observations)]
    n = len(obs)

    # 初始化
    mu = np.array([np.percentile(obs, 33), np.percentile(obs, 67)])
    sigma = np.array([np.std(obs) * 0.8, np.std(obs) * 1.2])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])  # 转移矩阵
    pi = np.array([0.5, 0.5])

    for _ in range(n_iter):
        # E-step: forward-backward
        # Forward
        alpha = np.zeros((n, 2))
        for j in range(2):
            alpha[0, j] = pi[j] * norm.pdf(obs[0], mu[j], sigma[j])
        alpha[0] /= alpha[0].sum()

        for t in range(1, n):
            for j in range(2):
                alpha[t, j] = norm.pdf(obs[t], mu[j], sigma[j]) * (alpha[t-1] @ A[:, j])
            alpha[t] /= alpha[t].sum()

        # Backward
        beta = np.zeros((n, 2))
        beta[-1] = 1.0
        for t in range(n-2, -1, -1):
            for j in range(2):
                beta[t, j] = (A[j] * norm.pdf(obs[t+1], mu, sigma) * beta[t+1]).sum()
            beta[t] /= beta[t].sum()

        # Gamma (state probabilities)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Xi (joint probabilities)
        xi = np.zeros((n-1, 2, 2))
        for t in range(n-1):
            for i in range(2):
                for j in range(2):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * norm.pdf(obs[t+1], mu[j], sigma[j]) * beta[t+1, j]
            xi[t] /= xi[t].sum()

        # M-step
        pi = gamma[0]
        for i in range(2):
            for j in range(2):
                A[i, j] = xi[:, i, j].sum() / gamma[:-1, i].sum()

        for j in range(2):
            w = gamma[:, j]
            mu[j] = (w * obs).sum() / w.sum()
            sigma[j] = np.sqrt((w * (obs - mu[j])**2).sum() / w.sum())
            sigma[j] = max(sigma[j], 1e-6)

    # 返回全长度(含NaN)的状态概率
    full_gamma = np.zeros((len(observations), 2))
    full_gamma[:] = np.nan
    valid_idx = ~np.isnan(observations)
    full_gamma[valid_idx] = gamma
    return full_gamma


def simulate_regime(px, sig, params):
    dates = px.index
    P_base = params.get("P", 3)
    k_base = params.get("k", 1.0)
    T = params.get("T", 90 * 24)
    use_flow = params.get("use_flow", True)
    regime_mode = params.get("regime_mode", "none")  # none / vol / hmm / perf
    W_hours = params.get("W_hours", 32 * 24)

    cash = INIT_CAPITAL
    pos = {}
    pend_buy, pend_sell = [], []
    trades = []
    equity = []

    # 准备状态信号
    if regime_mode == "vol":
        vol_ratio = sig.get("vol_ratio", None)
    elif regime_mode == "hmm":
        state_prob = sig.get("hmm_state", None)
    elif regime_mode == "perf":
        pass  # 动态计算

    warmup = W_hours + 200
    start_i = warmup

    # 性能追踪(for perf mode)
    recent_trades = []

    for i in range(start_i, len(dates)):
        today = px.iloc[i]
        today_z = sig["z"].iloc[i]

        # 确定当前状态下的参数
        k = k_base
        P = P_base

        if regime_mode == "vol" and vol_ratio is not None:
            # 取所有股票的vol_ratio中位数作为当前市场波动率状态
            vr = vol_ratio.iloc[i].median()
            if not pd.isna(vr):
                if vr < 0.8:  # 低波动 → 激进
                    k = k_base - 0.2
                    P = P_base + 1
                elif vr > 1.3:  # 高波动 → 保守
                    k = k_base + 0.3
                    P = max(P_base - 1, 1)

        elif regime_mode == "hmm" and state_prob is not None:
            prob_s1 = state_prob.iloc[i] if i < len(state_prob) else np.nan
            if not np.isnan(prob_s1):
                if prob_s1 > 0.6:  # 高回复状态 → 激进
                    k = k_base - 0.15
                elif prob_s1 < 0.4:  # 低回复状态 → 保守
                    k = k_base + 0.3

        elif regime_mode == "perf":
            if len(recent_trades) >= 10:
                recent_win_rate = sum(1 for t in recent_trades[-10:] if t > 0) / 10
                if recent_win_rate > 0.9:
                    k = k_base - 0.2
                    P = P_base + 1
                elif recent_win_rate < 0.7:
                    k = k_base + 0.3
                    P = max(P_base - 1, 1)

        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                net_r = today[sym] * (1 - SELL_TAX) / pos[sym]["entry_px"] - 1
                trades.append({
                    "stock": sym, "entry_date": dates[pos[sym]["entry_i"]],
                    "exit_date": dates[i], "net_ret": net_r,
                    "hold_days": (i - pos[sym]["entry_i"]) / 24,
                    "regime_k": pos[sym].get("regime_k", k_base),
                })
                if regime_mode == "perf":
                    recent_trades.append(net_r)
                del pos[sym]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                alloc = cash / (n_slots - j)
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i, "regime_k": k}
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
    eq, trades = simulate_regime(px, sig, params)
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
    W_hours = 32 * 24
    all_rows = []

    # 基础信号
    sig = {}
    sig["z"] = (px - px.rolling(W_hours).mean()) / px.rolling(W_hours).std()
    sig["flow"] = sh.pct_change(5 * 24)

    # 波动率状态信号
    vol_20 = px.pct_change().rolling(20 * 24).std()
    vol_200_med = px.pct_change().rolling(200 * 24).std().expanding().median()
    sig["vol_ratio"] = vol_20 / vol_200_med.replace(0, np.nan)

    # HMM状态: 用每日平均z-score拟合 (降采样到日频减少计算量)
    print("拟合HMM...")
    z_daily_mean = sig["z"].resample("1D").mean().mean(axis=1)  # 所有股票z的日均值
    hmm_states = simple_hmm_2state(z_daily_mean.values)
    # 上采样回小时频
    hmm_hourly = pd.Series(index=sig["z"].index, dtype=float)
    for i, (dt, val) in enumerate(zip(z_daily_mean.index, hmm_states[:, 1])):
        mask = (sig["z"].index >= dt) & (sig["z"].index < dt + pd.Timedelta(days=1))
        hmm_hourly[mask] = val
    sig["hmm_state"] = hmm_hourly

    # 分析HMM状态特征
    state1_mask = hmm_states[:, 1] > 0.5  # 高回复状态
    print(f"HMM状态: 状态1(高回复)={state1_mask.sum()}天, 状态0(低回复)={(~state1_mask).sum()}天")
    print(f"状态1占比: {state1_mask.mean()*100:.1f}%")

    # ── 基准 ──
    for label, params in [
        ("BASE_W32_k1.0_P3_f", {"W_hours": W_hours, "k": 1.0, "T": 90 * 24, "P": 3, "use_flow": True, "regime_mode": "none"}),
    ]:
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── A. 波动率分位法 ──
    for k in [1.0, 1.2]:
        label = f"vol_regime_k{k}"
        params = {"W_hours": W_hours, "k": k, "T": 90 * 24, "P": 3, "use_flow": True, "regime_mode": "vol"}
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── B. HMM状态法 ──
    for k in [1.0, 1.2]:
        label = f"hmm_regime_k{k}"
        params = {"W_hours": W_hours, "k": k, "T": 90 * 24, "P": 3, "use_flow": True, "regime_mode": "hmm"}
        rows = run_config(px, sig, params, label)
        all_rows.extend(rows)
        print(f"完成 {label}")

    # ── C. 性能反馈法 ──
    for k in [1.0]:
        label = f"perf_regime_k{k}"
        params = {"W_hours": W_hours, "k": k, "T": 90 * 24, "P": 3, "use_flow": True, "regime_mode": "perf"}
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

    print("\n=== HMM/状态切换实验 ===")
    pd.set_option("display.width", 400)
    print(view.sort_values("cagr", ascending=False).round(4).to_string())

    base_cagr = full.loc["BASE_W32_k1.0_P3_f", "cagr"]
    print(f"\n基准: {base_cagr*100:.1f}%")
    for cfg in view.sort_values("cagr", ascending=False).index:
        if cfg.startswith("BASE"):
            continue
        d = view.loc[cfg, "cagr"] - base_cagr
        marker = "★" if d > 0.005 else ("✓" if d > 0 else "✗")
        print(f"  {marker} {cfg}: {view.loc[cfg, 'cagr']*100:.1f}% (Δ={d*100:+.1f}pp)  | trades={view.loc[cfg, 'n_trades']:.0f}")

    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
