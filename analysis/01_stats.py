"""01_stats.py — 价格统计特征分析。

输出到 analysis/output/stats/:
  baseline.csv            每标的 3 年 CAGR / 年化波动 / 最大回撤 / 起止价
  autocorr_{h}.csv        各周期收益率 lag-1..20 自相关(行=标的)
  seasonality_hourly.csv  全市场按 UTC 小时的平均收益/波动(按标的标准化后汇总)
  seasonality_weekday.csv 全市场按星期几的平均收益/波动 + 周末 vs 工作日对比
  *.png                   自相关与季节性图

用法: .venv/bin/python analysis/01_stats.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DATA_DIR, OUTPUT_DIR,  # noqa: F401  (供 IDE/调试参考)
    cagr, ensure_out, list_stocks, load_stock, max_drawdown, resample_close,
)

HORIZONS = {"1min": "1min", "5min": "5min", "15min": "15min",
            "1h": "1h", "4h": "4h", "1D": "1D"}
N_LAGS = 20


def autocorr_table(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """每个周期一张表:行=标的,列=lag1..lagN 的自相关。"""
    tables = {name: {} for name in HORIZONS}
    for sym in stocks:
        df = load_stock(sym)
        for name, rule in HORIZONS.items():
            close = resample_close(df, rule)
            r = close.pct_change().dropna()
            tables[name][sym] = {f"lag{k}": r.autocorr(k) for k in range(1, N_LAGS + 1)}
    return {name: pd.DataFrame(rows).T for name, rows in tables.items()}


def baseline_table(stocks: list[str]) -> pd.DataFrame:
    rows = {}
    for sym in stocks:
        df = load_stock(sym)
        daily = resample_close(df, "1D")
        dr = daily.pct_change().dropna()
        rows[sym] = {
            "price_start": df["price"].iloc[0],
            "price_end": df["price"].iloc[-1],
            "cagr": cagr(daily),
            "ann_vol": dr.std() * np.sqrt(365.25),
            "max_drawdown": max_drawdown(daily),
            "daily_ret_kurtosis": dr.kurtosis(),
        }
    return pd.DataFrame(rows).T


def seasonality(stocks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按小时/星期几的全市场季节性。

    对每只股票:1h 收益率 → 减去自身均值(去趋势),按 UTC 小时/星期几分组。
    再对所有股票取平均,得到市场层面形状。
    """
    hourly_ret, hourly_vol = [], []
    wday_ret, wday_vol = [], []
    for sym in stocks:
        if sym == "TCSE":
            continue
        close = resample_close(load_stock(sym), "1h")
        r = close.pct_change().dropna()
        r = r - r.mean()  # 去掉个股漂移,只留季节性形状
        hourly_ret.append(r.groupby(r.index.hour).mean())
        hourly_vol.append(r.groupby(r.index.hour).std())
        wday_ret.append(r.groupby(r.index.dayofweek).mean())
        wday_vol.append(r.groupby(r.index.dayofweek).std())

    hourly = pd.DataFrame({
        "mean_ret_bp": pd.concat(hourly_ret, axis=1).mean(axis=1) * 1e4,
        "vol_bp": pd.concat(hourly_vol, axis=1).mean(axis=1) * 1e4,
    })
    hourly.index.name = "utc_hour"

    wday = pd.DataFrame({
        "mean_ret_bp": pd.concat(wday_ret, axis=1).mean(axis=1) * 1e4,
        "vol_bp": pd.concat(wday_vol, axis=1).mean(axis=1) * 1e4,
    })
    wday.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wday.index.name = "weekday"

    weekend_vol = wday.loc[["Sat", "Sun"], "vol_bp"].mean()
    weekday_vol = wday.loc[["Mon", "Tue", "Wed", "Thu", "Fri"], "vol_bp"].mean()
    wday.loc["weekday_avg"] = [np.nan, weekday_vol]
    wday.loc["weekend_avg"] = [np.nan, weekend_vol]
    wday.loc["ratio_wd_we"] = [np.nan, weekday_vol / weekend_vol]
    return hourly, wday


def make_plots(ac: dict[str, pd.DataFrame], hourly: pd.DataFrame,
               wday: pd.DataFrame, out) -> None:
    # 1) 各周期 lag1..5 自相关的全市场均值
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, tbl in ac.items():
        cols = [f"lag{k}" for k in range(1, 6)]
        ax.plot(range(1, 6), tbl[cols].mean().values, marker="o", label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("mean autocorr (35 stocks)")
    ax.set_title("Return autocorrelation by horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "autocorr_by_horizon.png", dpi=120)
    plt.close(fig)

    # 2) 日内波动曲线
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly.index, hourly["vol_bp"], marker="o")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("1h return vol (bp)")
    ax.set_title("Intraday volatility (market average)")
    fig.tight_layout()
    fig.savefig(out / "hourly_vol.png", dpi=120)
    plt.close(fig)

    # 3) 周内波动
    fig, ax = plt.subplots(figsize=(8, 4))
    wd = wday.loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
    ax.bar(wd.index, wd["vol_bp"])
    ax.set_ylabel("1h return vol (bp)")
    ax.set_title("Weekday volatility (market average)")
    fig.tight_layout()
    fig.savefig(out / "weekday_vol.png", dpi=120)
    plt.close(fig)


def main() -> None:
    out = ensure_out("stats")
    stocks = list_stocks()
    print(f"标的数: {len(stocks)}")

    base = baseline_table(stocks)
    base.to_csv(out / "baseline.csv")
    print("\n=== 基准(3 年)===")
    print(base.round(4).to_string())

    ac = autocorr_table(stocks)
    for name, tbl in ac.items():
        tbl.to_csv(out / f"autocorr_{name.replace('min', 'm')}.csv")
        sig = 1.96 / np.sqrt(len(tbl))  # 跨标的均值的粗略显著界
        mean_ac = tbl.mean()
        print(f"\n=== {name} 收益率自相关(35 股均值,|rho|>{sig:.4f} 约显著)===")
        print(mean_ac.round(4).head(8).to_string())

    hourly, wday = seasonality(stocks)
    hourly.to_csv(out / "seasonality_hourly.csv")
    wday.to_csv(out / "seasonality_weekday.csv")
    print("\n=== 按星期几(全市场 1h 收益,单位 bp)===")
    print(wday.round(3).to_string())
    print("\n=== 按 UTC 小时(前 5 行,完整见 CSV)===")
    print(hourly.round(3).head().to_string())

    make_plots(ac, hourly, wday, out)
    print(f"\n输出目录: {out}")


if __name__ == "__main__":
    main()
