"""exp_alpha_stack.py — 实验3: Alpha残差正交化 + 弱信号堆叠。

核心问题: z_score之外是否存在独立于z的第二维可预测信号?
之前ML的结论是"没有", 但可能只是因为GBM没正确使用弱信号。

方法:
1. 对每个弱信号做对z_30的正交回归, 提取残差
2. 计算残差对未来收益的IC (独立于z的预测力)
3. 如果残差IC显著 → 构建正交化堆叠信号
4. 如果残差IC不显著 → 确认"只有一维信号"

弱信号池:
- dshares_5d (资金流变化, IC=+0.12)
- ret_1d / ret_5d (短期反转)
- dd_20 (距20日高点)
- rsi_14
- range_pred (振幅预测排名)
- vol_ratio (波动率状态)

用法: .venv/bin/python analysis/exp_alpha_stack.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_alpha")
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


def compute_factors(px, sh):
    """计算全部候选因子"""
    f = {}
    ret = px.pct_change()

    # z_score (主因子)
    f["z_30"] = (px - px.rolling(30).mean()) / px.rolling(30).std()
    f["z_20"] = (px - px.rolling(20).mean()) / px.rolling(20).std()

    # 弱信号
    f["dshares_5d"] = sh.pct_change(5)                 # 资金流
    f["ret_1d"] = ret                                   # 1日反转
    f["ret_5d"] = px.pct_change(5)                     # 5日反转
    f["ret_10d"] = px.pct_change(10)                   # 10日反转
    f["dd_20"] = px / px.rolling(20).max() - 1          # 距高点跌幅
    f["rsi_14"] = compute_rsi(px, 14)                  # RSI
    f["vol_20"] = ret.rolling(20).std()                 # 波动率

    # 振幅预测 (过去k天平均日内振幅)
    f["range_5d"] = compute_range_pred(px, 5)
    f["range_10d"] = compute_range_pred(px, 10)

    # 波动率比
    vol_200_med = ret.rolling(200).std().expanding().median()
    f["vol_ratio"] = f["vol_20"] / vol_200_med.replace(0, np.nan)

    return f, ret


def compute_rsi(px, window):
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_range_pred(px, window):
    """过去k天平均日内振幅(用日线OHLC模拟: 用前后日收盘差+日内波动代理)"""
    # 简化:用过去k天绝对收益之和/k作为振幅代理
    return px.diff().abs().rolling(window).mean()


def orthogonalize_residuals(f, target_factor="z_30"):
    """对所有弱信号做对z_30的正交回归, 返回残差矩阵。"""
    z = f[target_factor]
    residuals = {}
    weak_signals = [k for k in f if k != target_factor and k != "z_20"]
    for name in weak_signals:
        sig = f[name]
        # 在横截面上(每一天)做回归: sig_i = a + b*z_i + eps_i
        res = pd.DataFrame(index=z.index, columns=z.columns, dtype=float)
        for t_idx in range(len(z.index)):
            z_t = z.iloc[t_idx]
            s_t = sig.iloc[t_idx]
            valid = z_t.notna() & s_t.notna()
            if valid.sum() < 10:
                continue
            slope, intercept, _, _, _ = stats.linregress(z_t[valid].values, s_t[valid].values)
            res.iloc[t_idx] = s_t - (intercept + slope * z_t)
        residuals[name] = res
    return residuals


def compute_ic(factor_df, fwd_ret, horizon=7):
    """计算横截面Spearman IC (每个时间点, 多只股票的因子值与未来收益的秩相关)。"""
    ic_list = []
    for i in range(len(factor_df.index) - horizon):
        f_t = factor_df.iloc[i]
        r_t = fwd_ret.iloc[i + horizon]  # 前瞻收益
        valid = f_t.notna() & r_t.notna()
        if valid.sum() < 10:
            continue
        ic, _ = stats.spearmanr(f_t[valid].values, r_t[valid].values)
        if not np.isnan(ic):
            ic_list.append(ic)
    if not ic_list:
        return 0, 0
    arr = np.array(ic_list)
    mean_ic = arr.mean()
    t_stat = mean_ic / (arr.std() / np.sqrt(len(arr))) if arr.std() > 0 else 0
    return mean_ic, t_stat


def compute_fwd_returns(px, horizons):
    """计算多个前瞻窗口的收益。"""
    fwd = {}
    for h in horizons:
        fwd[h] = px.shift(-h) / px - 1
    return fwd


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    px, sh = load_panel()
    f, ret = compute_factors(px, sh)

    # 只使用train期做分析 (避免未来函数)
    train_mask = px.index < SPLIT_TRAIN_END
    px_train = px[train_mask]
    f_train = {k: v[train_mask] for k, v in f.items()}

    # 前瞻收益
    horizons = [3, 7, 14, 30]
    fwd = compute_fwd_returns(px_train, horizons)

    print("=" * 80)
    print("Part 1: 各因子的原始IC (train期内)")
    print("=" * 80)
    ic_results = []
    for name in sorted(f_train.keys()):
        for h in horizons:
            mean_ic, t_stat = compute_ic(f_train[name], fwd[h], h)
            ic_results.append({"factor": name, "horizon": h,
                               "IC": mean_ic, "t": t_stat})
    ic_df = pd.DataFrame(ic_results)
    ic_pivot = ic_df.pivot_table(values=["IC", "t"], index="factor", columns="horizon")
    print(ic_pivot.round(4).to_string())

    # 关键:残差IC分析
    print("\n" + "=" * 80)
    print("Part 2: 残差IC分析 (对z_30正交化后, 残差信号是否有独立预测力?)")
    print("=" * 80)
    residuals = orthogonalize_residuals(f_train, "z_30")
    res_ic = []
    for name, res_df in sorted(residuals.items()):
        for h in horizons:
            mean_ic, t_stat = compute_ic(res_df, fwd[h], h)
            # 找出原始IC对比
            orig_ic, orig_t = compute_ic(f_train[name], fwd[h], h)
            res_ic.append({"factor": name, "horizon": h,
                           "orig_IC": orig_ic, "orig_t": orig_t,
                           "resid_IC": mean_ic, "resid_t": t_stat,
                           "delta": mean_ic - orig_ic})
    res_df = pd.DataFrame(res_ic)
    print("\n--- 14日前瞻 (核心判断窗口) ---")
    res_14 = res_df[res_df.horizon == 14].set_index("factor")
    res_14["independent?"] = res_14.apply(
        lambda r: "YES ★" if abs(r.resid_t) > 2.0 else "NO", axis=1)
    print(res_14[["orig_IC", "orig_t", "resid_IC", "resid_t", "independent?"]].round(4).to_string())

    # 全面分析
    print("\n--- 7日前瞻 ---")
    res_7 = res_df[res_df.horizon == 7].set_index("factor")
    res_7["independent?"] = res_7.apply(
        lambda r: "YES ★" if abs(r.resid_t) > 2.0 else "NO", axis=1)
    print(res_7[["orig_IC", "orig_t", "resid_IC", "resid_t", "independent?"]].round(4).to_string())

    # 判断
    print("\n" + "=" * 80)
    print("Part 3: 结论")
    print("=" * 80)
    independent_signals = res_14[res_14["independent?"] == "YES ★"]
    if len(independent_signals) > 0:
        print(f"发现 {len(independent_signals)} 个独立于z_30的信号 (|t|>2):")
        print(independent_signals.to_string())
        print("\n→ 原报告'ML无增益'的结论需要修正: 存在第二维可预测信号!")
    else:
        print("所有弱信号的残差IC在控制z_30后均不显著 (|t|<2)。")
        print("→ 确认原报告结论: 可预测结构只有'偏离均线'这一个维度。")
        print("→ 继续用ML/非线性方法搜索隐藏信号的意义不大。")

    # 但: 如果残差IC在方向上是正确的(虽然不显著), 也许可以用ensemble方法
    print("\n" + "=" * 80)
    print("Part 4: 残差信号方向一致性检查")
    print("=" * 80)
    for name in sorted(residuals.keys()):
        res_14_ic = res_14.loc[name, "resid_IC"] if name in res_14.index else 0
        orig_14_ic = res_14.loc[name, "orig_IC"] if name in res_14.index else 0
        # 残差IC与原始IC同号 = 残差保留了部分预测方向
        if abs(orig_14_ic) > 0.01 and abs(res_14_ic) > 0.005:
            direction = "同向" if orig_14_ic * res_14_ic > 0 else "反向"
        else:
            direction = "噪声"
        print(f"  {name:20s}: 原始IC={orig_14_ic:+.4f}, 残差IC={res_14_ic:+.4f} → {direction}")

    print(f"\n输出目录: {OUTDIR}")


if __name__ == "__main__":
    main()
