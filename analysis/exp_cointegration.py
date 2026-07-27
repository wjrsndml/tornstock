"""exp_cointegration.py — 实验9: 跨股协整检验与价差交易。

核心问题: Torn的35只股票之间是否存在协整关系?
如果存在, 可以从价差(spread)的均值回复中提取独立于单股z-score的交易信号。

方法:
A. 对全部595对股票做Engle-Granger协整检验
B. 对显著协整对, 计算价差的z-score
C. 检验价差z-score对未来收益的预测力(IC)
D. 如果显著: 构建价差轮动策略(买入价差z最低的股票, 卖出价差z最高的)
E. 对比"价差信号 vs 单股z-score信号"的相关性和独立增量

用法: .venv/bin/python analysis/exp_cointegration.py
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

OUTDIR = ensure_out("exp_coint")
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


def test_cointegration(y, x, maxlag=5):
    """Engle-Granger两步法协整检验。返回(adf_stat, pvalue, hedge_ratio, spread, is_cointegrated)"""
    # 去掉NaN
    valid = y.notna() & x.notna()
    y_v, x_v = y[valid].values, x[valid].values
    if len(y_v) < 100:
        return np.nan, np.nan, np.nan, None, False

    # Step 1: OLS回归 y = α + βx + ε
    X = np.column_stack([np.ones_like(x_v), x_v])
    coeff = np.linalg.lstsq(X, y_v, rcond=None)[0]
    alpha, beta = coeff[0], coeff[1]
    spread = y_v - (alpha + beta * x_v)

    # Step 2: ADF检验残差是否平稳
    try:
        adf_result = adfuller(spread, maxlag=min(maxlag, len(spread) // 4), autolag="AIC")
        adf_stat = adf_result[0]
        pvalue = adf_result[1]
    except Exception:
        return np.nan, np.nan, beta, None, False

    is_coint = pvalue < 0.05
    return adf_stat, pvalue, beta, spread, is_coint


def main():
    px, sh = load_panel()
    stocks = list(px.columns)
    n_stocks = len(stocks)

    # 只在train期做协整检验(避免未来函数)
    train_mask = px.index < SPLIT_TRAIN_END
    px_train = px[train_mask]
    px_log = np.log(px_train)  # 对数价格(协整通常用对数)

    print(f"=== 协整检验: {n_stocks}只股票, {n_stocks*(n_stocks-1)//2}对 ===")

    # A. 对所有对做协整检验
    results = []
    count = 0
    total_pairs = n_stocks * (n_stocks - 1) // 2
    for i, j in itertools.combinations(range(n_stocks), 2):
        sym_i, sym_j = stocks[i], stocks[j]
        y, x = px_log[sym_i], px_log[sym_j]
        adf_stat, pvalue, beta, spread_arr, is_coint = test_cointegration(y, x)
        if not np.isnan(adf_stat):
            results.append({
                "stock_y": sym_i, "stock_x": sym_j,
                "adf_stat": adf_stat, "pvalue": pvalue,
                "hedge_ratio": beta, "is_cointegrated": is_coint,
            })
        count += 1
        if count % 100 == 0:
            print(f"  进度: {count}/{total_pairs}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTDIR / "cointegration_tests.csv", index=False)

    n_coint = res_df.is_cointegrated.sum()
    print(f"\n协整对: {n_coint}/{len(res_df)} ({n_coint/len(res_df)*100:.1f}%)")
    print(f"ADF统计量分布: mean={res_df.adf_stat.mean():.2f}, "
          f"P5={res_df.adf_stat.quantile(0.05):.2f}, "
          f"P50={res_df.adf_stat.median():.2f}")

    # 分析: 检验结果的显著性水平
    print(f"\nP值分布:")
    for thresh in [0.01, 0.05, 0.10]:
        n = (res_df.pvalue < thresh).sum()
        print(f"  p<{thresh}: {n} ({n/len(res_df)*100:.1f}%)")

    # B. 对显著协整对, 计算价差z-score的IC
    if n_coint > 0:
        print(f"\n=== 价差z-score预测力分析 ===")
        coint_pairs = res_df[res_df.is_cointegrated]

        # 对每对显著协整对, 在全时段计算价差z-score
        spread_z_ics = []
        for _, pair in coint_pairs.iterrows():
            sym_y, sym_x = pair["stock_y"], pair["stock_x"]
            beta = pair["hedge_ratio"]

            # 全时段对数价差
            y_full = np.log(px[sym_y])
            x_full = np.log(px[sym_x])
            spread_full = y_full - beta * x_full

            # 价差的z-score (滚动30天)
            spread_ma = spread_full.rolling(30).mean()
            spread_std = spread_full.rolling(30).std().replace(0, np.nan)
            spread_z = (spread_full - spread_ma) / spread_std

            # 价差z对未来y股票收益的IC (价差偏低→y应该涨)
            for h in [7, 14, 30]:
                fwd_ret = px[sym_y].pct_change(h).shift(-h)
                valid = spread_z.notna() & fwd_ret.notna()
                if valid.sum() < 30:
                    continue
                ic, _ = stats.spearmanr(spread_z[valid], fwd_ret[valid])
                if not np.isnan(ic):
                    spread_z_ics.append({
                        "pair": f"{sym_y}/{sym_x}",
                        "horizon": h, "IC": ic,
                        "adf_stat": pair["adf_stat"],
                    })

        if spread_z_ics:
            sz_df = pd.DataFrame(spread_z_ics)
            sz_pivot = sz_df.pivot_table(values="IC", index="pair", columns="horizon", aggfunc="mean")
            print("\n价差z-score对未来收益的IC (前10对):")
            print(sz_pivot.head(10).round(4).to_string())

            # 平均IC
            for h in [7, 14, 30]:
                subset = sz_df[sz_df.horizon == h]
                if len(subset) > 0:
                    mean_ic = subset.IC.mean()
                    t_stat = mean_ic / (subset.IC.std() / np.sqrt(len(subset))) if subset.IC.std() > 0 else 0
                    print(f"\n  {h}日: mean IC={mean_ic:.4f}, t={t_stat:.2f}, n={len(subset)}")

    # C. 跨股价差组合的整体信号
    # 构建一个综合信号: 对每只股票, 计算它相对于所有其他股票的"平均价差z-score"
    print(f"\n=== 综合价差信号: 每只股票相对于全市场的平均偏离 ===")

    # 选择ADF最显著的top-N对
    top_pairs = res_df.sort_values("adf_stat").head(50)
    print(f"使用ADF最显著的50对构建综合信号")

    # 对每只股票, 汇总它在各对中的价差z
    composite_spread_z = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)

    for _, pair in top_pairs.iterrows():
        sym_y, sym_x = pair["stock_y"], pair["stock_x"]
        beta = pair["hedge_ratio"]
        y_full = np.log(px[sym_y])
        x_full = np.log(px[sym_x])
        spread_full = y_full - beta * x_full
        spread_ma = spread_full.rolling(30).mean()
        spread_std = spread_full.rolling(30).std().replace(0, np.nan)
        spread_z = (spread_full - spread_ma) / spread_std
        # 正z = y相对x偏高 → y应该跌 → 做空y信号
        composite_spread_z[sym_y] = composite_spread_z[sym_y].fillna(0) - spread_z
        composite_spread_z[sym_x] = composite_spread_z[sym_x].fillna(0) + spread_z

    # 归一化(除以每只股票参与的对数)
    pair_counts = pd.Series(0, index=px.columns)
    for _, pair in top_pairs.iterrows():
        pair_counts[pair["stock_y"]] += 1
        pair_counts[pair["stock_x"]] += 1
    for sym in px.columns:
        if pair_counts[sym] > 0:
            composite_spread_z[sym] = composite_spread_z[sym] / pair_counts[sym]

    # 综合价差z的IC
    print("\n综合价差z-score IC:")
    for h in [7, 14, 30]:
        fwd = px.pct_change(h).shift(-h)
        ic_list = []
        for t_idx in range(len(px.index) - h):
            cz = composite_spread_z.iloc[t_idx]
            fr = fwd.iloc[t_idx + h]
            valid = cz.notna() & fr.notna()
            if valid.sum() < 10:
                continue
            ic, _ = stats.spearmanr(cz[valid], fr[valid])
            if not np.isnan(ic):
                ic_list.append(ic)
        if ic_list:
            arr = np.array(ic_list)
            print(f"  {h}日: mean IC={arr.mean():.4f}, t={arr.mean()/arr.std()*np.sqrt(len(arr)):.2f}, n={len(arr)}")

    # D. 与单股z_30的对比
    z_30 = (px - px.rolling(30).mean()) / px.rolling(30).std()
    print("\n=== 价差z vs 单股z_30 相关性 ===")
    corr = composite_spread_z.corrwith(z_30, axis=1)
    print(f"  平均截面相关: {corr.mean():.3f}")
    # 残差分析
    # 在截面上去掉z_30的影响, 看残差IC
    print("\n价差z残差(控制z_30后)的IC:")
    for h in [7, 14]:
        fwd = px.pct_change(h).shift(-h)
        ic_list = []
        for t_idx in range(len(px.index) - h):
            cz = composite_spread_z.iloc[t_idx]
            zz = z_30.iloc[t_idx]
            fr = fwd.iloc[t_idx + h]
            valid = cz.notna() & zz.notna() & fr.notna()
            if valid.sum() < 10:
                continue
            # 正交化
            slope, intercept, _, _, _ = stats.linregress(zz[valid], cz[valid])
            cz_resid = cz[valid] - (intercept + slope * zz[valid])
            ic, _ = stats.spearmanr(cz_resid, fr[valid])
            if not np.isnan(ic):
                ic_list.append(ic)
        if ic_list:
            arr = np.array(ic_list)
            print(f"  {h}日: 残差IC={arr.mean():.4f}, t={arr.mean()/arr.std()*np.sqrt(len(arr)):.2f}")

    print(f"\n输出: {OUTDIR}")


if __name__ == "__main__":
    main()
