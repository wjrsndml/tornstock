"""exp_leadlag.py — 实验10: 跨股领先-滞后网络。

方法:
A. 对所有有序对(i→j)计算滞后交叉相关: corr(ret_i(t), ret_j(t+k)), k=1..10天
B. 筛选显著领先-滞后对(|corr|>阈值, |t|>阈值)
C. 构建"领先网络信号": 如果i是j的领先者且i最近暴跌, 则j预计也会跌→推迟买入j
D. 检验领先信号对策略的边际贡献

用法: .venv/bin/python analysis/exp_leadlag.py
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    SPLIT_TRAIN_END, ensure_out, list_stocks, load_stock, resample_close, cagr,
)

OUTDIR = ensure_out("exp_leadlag")


def load_panel():
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes = {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1D")
    return pd.DataFrame(closes).ffill()


def compute_leadlag(px, max_lag=10):
    """对所有有序对计算滞后交叉相关。

    返回: DataFrame with columns [leader, follower, lag, corr, t_stat]
    """
    ret = px.pct_change().dropna()
    stocks = list(px.columns)
    results = []

    for leader, follower in itertools.permutations(stocks, 2):
        r_l = ret[leader].dropna()
        r_f = ret[follower].dropna()
        common_idx = r_l.index.intersection(r_f.index)
        if len(common_idx) < 200:
            continue
        r_l = r_l[common_idx]
        r_f = r_f[common_idx]

        for lag in range(1, max_lag + 1):
            # corr(r_l(t), r_f(t+lag))
            r_l_aligned = r_l.iloc[:-lag]
            r_f_aligned = r_f.iloc[lag:]
            if len(r_l_aligned) < 100:
                continue
            corr_val = r_l_aligned.corr(r_f_aligned)
            if np.isnan(corr_val):
                continue
            # t-statistic
            n = len(r_l_aligned)
            t_stat = corr_val * np.sqrt((n - 2) / (1 - corr_val ** 2)) if abs(corr_val) < 1 else np.inf
            results.append({
                "leader": leader,
                "follower": follower,
                "lag": lag,
                "corr": corr_val,
                "t_stat": t_stat,
                "n": n,
            })

    return pd.DataFrame(results)


def main():
    px = load_panel()

    # 只在train期做分析
    train_mask = px.index < SPLIT_TRAIN_END
    px_train = px[train_mask]

    print("=== 计算领先-滞后网络 ===")
    ll_df = compute_leadlag(px_train, max_lag=10)
    total = len(ll_df)
    print(f"  共 {total} 个有序对 × 滞后组合")

    # 显著性筛选
    sig_pairs = ll_df[(ll_df["t_stat"].abs() > 2.0) & (ll_df["corr"].abs() > 0.03)]
    print(f"  显著对 (|t|>2, |corr|>0.03): {len(sig_pairs)}")

    # 分析
    print(f"\n=== 领先-滞后统计 ===")
    print(f"corr分布: mean={ll_df['corr'].mean():.4f}, std={ll_df['corr'].std():.4f}")
    print(f"  P95={ll_df['corr'].quantile(0.95):.4f}, P99={ll_df['corr'].quantile(0.99):.4f}")
    print(f"  max={ll_df['corr'].max():.4f}, min={ll_df['corr'].min():.4f}")

    # Top显著对
    top_sig = sig_pairs.sort_values("corr", ascending=False).head(20)
    print(f"\nTop-20 领先-滞后对:")
    print(top_sig[["leader", "follower", "lag", "corr", "t_stat"]].to_string(index=False))

    # 聚合:每只股票作为"领先者"的显著对数量
    leader_counts = sig_pairs.groupby("leader").size().sort_values(ascending=False)
    follower_counts = sig_pairs.groupby("follower").size().sort_values(ascending=False)
    print(f"\n最多领先者(top-5): {dict(leader_counts.head(5))}")
    print(f"最多跟随者(top-5): {dict(follower_counts.head(5))}")

    # 关键检验: 领先者收益能否预测跟随者收益?
    # 对每个显著对, 在test期验证
    print(f"\n=== 样本外验证: 领先信号能否预测跟随者? ===")
    px_test = px[px.index >= SPLIT_TRAIN_END]
    ret_test = px_test.pct_change().dropna()

    top_pairs = sig_pairs.sort_values("corr", ascending=False).head(30)
    oos_results = []
    for _, pair in top_pairs.iterrows():
        leader = pair["leader"]
        follower = pair["follower"]
        lag = int(pair["lag"])
        if leader not in ret_test.columns or follower not in ret_test.columns:
            continue
        r_l = ret_test[leader].dropna()
        r_f = ret_test[follower].dropna()
        common = r_l.index.intersection(r_f.index)
        if len(common) < 50:
            continue
        r_l_a = r_l[common].iloc[:-lag]
        r_f_a = r_f[common].iloc[lag:]
        if len(r_l_a) < 30:
            continue
        corr_oos = r_l_a.corr(r_f_a)
        t_oos = corr_oos * np.sqrt((len(r_l_a) - 2) / max(1 - corr_oos**2, 1e-6))
        oos_results.append({
            "leader": leader, "follower": follower,
            "lag": lag, "corr_is": pair["corr"], "corr_oos": corr_oos,
            "t_oos": t_oos,
        })

    oos_df = pd.DataFrame(oos_results)
    if len(oos_df) > 0:
        print(f"Top-30对样本外验证 (n={len(oos_df)}):")
        print(f"  IS corr: mean={oos_df.corr_is.mean():.4f}, median={oos_df.corr_is.median():.4f}")
        print(f"  OOS corr: mean={oos_df.corr_oos.mean():.4f}, median={oos_df.corr_oos.median():.4f}")
        oos_sig = (oos_df.t_oos.abs() > 2).sum()
        print(f"  OOS显著 (|t|>2): {oos_sig}/{len(oos_df)} ({oos_sig/len(oos_df)*100:.0f}%)")

        # 方向一致性
        same_sign = ((oos_df.corr_is > 0) == (oos_df.corr_oos > 0)).mean()
        print(f"  IS/OOS方向一致: {same_sign*100:.0f}%")

    # 如果领先-滞后关系可靠, 构建复合领先信号
    # 对每只股票: "领先压力" = 它的领先者们的加权平均近期收益
    if oos_sig > 5:  # 至少5个显著对OOS仍有效
        print(f"\n=== 构建领先压力信号 ===")
        # 使用OOS显著的领先-滞后对
        valid_pairs = oos_df[oos_df.t_oos.abs() > 2]
        print(f"使用 {len(valid_pairs)} 个OOS显著的领先-滞后对")

        lead_pressure = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for _, pair in valid_pairs.iterrows():
            leader = pair["leader"]
            follower = pair["follower"]
            lag = int(pair["lag"])
            weight = pair["corr_oos"]
            # 领先者近期lag天收益 → 预测跟随者
            leader_ret = px[leader].pct_change(lag)
            # 正领先相关: 领先者涨→跟随者涨, 领先信号=正
            lead_pressure[follower] = lead_pressure[follower].fillna(0) + leader_ret * weight

        # 检验领先压力信号的IC
        for h in [3, 7]:
            fwd = px.pct_change(h).shift(-h)
            z30 = (px - px.rolling(30).mean()) / px.rolling(30).std()
            ic_list, resid_ic_list = [], []
            for t_idx in range(len(px.index) - h - 100):
                lp = lead_pressure.iloc[t_idx + 100]  # 跳过预热
                zz = z30.iloc[t_idx + 100]
                fr = fwd.iloc[t_idx + 100 + h]
                valid = lp.notna() & zz.notna() & fr.notna() & (lp.abs() > 1e-9)
                if valid.sum() < 10:
                    continue
                ic, _ = stats.spearmanr(lp[valid], fr[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
                # 残差
                slope, intercept, _, _, _ = stats.linregress(zz[valid], lp[valid])
                lp_resid = lp[valid] - (intercept + slope * zz[valid])
                ic_r, _ = stats.spearmanr(lp_resid, fr[valid])
                if not np.isnan(ic_r):
                    resid_ic_list.append(ic_r)
            if ic_list:
                arr = np.array(ic_list)
                print(f"  {h}日: 领先压力IC={arr.mean():.4f}, t={arr.mean()/arr.std()*np.sqrt(len(arr)):.2f}")
            if resid_ic_list:
                arr = np.array(resid_ic_list)
                print(f"  {h}日: 残差IC(控z_30)={arr.mean():.4f}, t={arr.mean()/arr.std()*np.sqrt(len(arr)):.2f}")
    else:
        print(f"\n领先-滞后关系在OOS中不成立 ({oos_sig} 个显著对), 不构建信号。")

    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
