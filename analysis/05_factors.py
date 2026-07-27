"""05_factors.py — 因子/指标 IC 研究:哪些指标对未来收益有预测力?

对每只股票(日线)计算一套指标,检验其与未来 1/3/7/14/30 日收益的
Spearman IC(按标的分别计算后取跨标的均值,t 值 = mean/SE)。

指标清单:
  动量/反转:  ret_5d, ret_10d, ret_20d, ret_60d
  趋势偏离:  z_20, z_60, z_120       (价格对 N 日均线的标准分)
  区间位置:  dd_20, dd_60, dd_120    (距 N 日最高点的跌幅,负值)
             ru_60                   (距 N 日最低点的涨幅,正值)
  技术指标:  rsi_14, boll_20         (布林带 %B)
  波动/偏度: vol_20, skew_20
  资金流:    dshares_5d, dshares_20d (total_shares 变动率)

输出 analysis/output/factors/ic_table.csv + 终端汇总。

用法: .venv/bin/python analysis/05_factors.py
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import ensure_out, list_stocks, load_stock, resample_close

FWD_DAYS = [1, 3, 7, 14, 30]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_factors(close: pd.Series, shares: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    for w in [5, 10, 20, 60]:
        f[f"ret_{w}d"] = close.pct_change(w)
    for w in [20, 60, 120]:
        ma, sd = close.rolling(w).mean(), close.rolling(w).std()
        f[f"z_{w}"] = (close - ma) / sd
    for w in [20, 60, 120]:
        f[f"dd_{w}"] = close / close.rolling(w).max() - 1
    f["ru_60"] = close / close.rolling(60).min() - 1
    f["rsi_14"] = rsi(close)
    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["boll_20"] = (close - (ma20 - 2 * sd20)) / (4 * sd20)
    r1 = close.pct_change()
    f["vol_20"] = r1.rolling(20).std()
    f["skew_20"] = r1.rolling(20).skew()
    for w in [5, 20]:
        f[f"dshares_{w}d"] = shares.pct_change(w)
    # 前瞻收益:次日收盘买入 → 持有 h 天
    for h in FWD_DAYS:
        f[f"fwd_{h}d"] = close.shift(-(h + 1)) / close.shift(-1) - 1
    return f


def main() -> None:
    outdir = ensure_out("factors")
    stocks = [s for s in list_stocks() if s != "TCSE"]

    factor_cols, ics = None, []
    for sym in stocks:
        df = load_stock(sym)
        close = resample_close(df, "1D")
        shares = df["total_shares"].resample("1D").last().reindex(close.index)
        f = compute_factors(close, shares).dropna()
        if factor_cols is None:
            factor_cols = [c for c in f.columns if not c.startswith("fwd_")]
        row = {}
        for fc in factor_cols:
            for h in FWD_DAYS:
                rho, _ = sps.spearmanr(f[fc], f[f"fwd_{h}d"])
                row[f"{fc}__{h}d"] = rho
        ics.append(row)

    ic = pd.DataFrame(ics, index=stocks)
    ic.to_csv(outdir / "ic_by_stock.csv")

    summary = pd.DataFrame({
        "IC_mean": ic.mean(),
        "IC_t": ic.mean() / (ic.std() / np.sqrt(len(ic))),
    })
    summary.index = pd.MultiIndex.from_tuples(
        [tuple(k.split("__")) for k in summary.index], names=["factor", "horizon"])
    tbl = summary.unstack("horizon")
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns)
    tbl.to_csv(outdir / "ic_table.csv")

    pd.set_option("display.width", 250)
    for metric in ["IC_mean", "IC_t"]:
        print(f"\n=== {metric}(行=因子,列=前瞻窗口)===")
        print(tbl[metric].round(3).to_string())

    # 按 |IC_mean| 排序的重点结论
    flat = summary.reindex(summary.IC_mean.abs().sort_values(ascending=False).index)
    print("\n=== |IC| Top12 因子-窗口 ===")
    print(flat.head(12).round(4).to_string())
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
