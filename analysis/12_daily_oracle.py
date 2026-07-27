"""12_daily_oracle.py — 日线级全知者:每天一次完美择时的收益上限(含未来函数)。

规则(按用户指定):
  - 以自然日(UTC)为窗口,每天最多操作一次;
  - 在 35 只股票中选出"当日振幅(高/低-1)最大、且最低点时间早于最高点"的那只;
  - 在当日最低价全仓买入,当日最高价卖出(扣 0.1% 卖出税);
  - 若选出的最大振幅 ≤ 0.1%,当天不操作;
  - 逐日复利,报告年化。

注意:这是含未来函数的理论上限,不可实盘,仅用于标定"日内择时"的天花板。

输出 analysis/output/oracle/daily_oracle.csv

用法: .venv/bin/python analysis/12_daily_oracle.py
"""

import numpy as np
import pandas as pd

from common import SELL_TAX, ensure_out, list_stocks, load_stock

MIN_RANGE = 0.001   # 振幅门槛 0.1%


def daily_extremes(sym: str) -> pd.DataFrame:
    """每股每日:最低价、最高价、各自首次出现的分钟。"""
    df = load_stock(sym)
    g = df.groupby(df.index.date)["price"]
    # 需要低/高点的时间先后,用 idxmin/idxmax(首次出现)
    low_t = df.groupby(df.index.date)["price"].idxmin()
    high_t = df.groupby(df.index.date)["price"].idxmax()
    out = pd.DataFrame({
        "low": g.min(), "high": g.max(),
        "low_t": low_t, "high_t": high_t,
    })
    out.index = pd.to_datetime(out.index)
    out["range"] = out["high"] / out["low"] - 1
    out["low_before_high"] = out["low_t"] < out["high_t"]
    out["stock"] = sym
    return out[["range", "low_before_high", "stock"]]


def main() -> None:
    outdir = ensure_out("oracle")
    stocks = [s for s in list_stocks() if s != "TCSE"]
    daily = pd.concat([daily_extremes(s) for s in stocks])

    days = []
    for date, g in daily.groupby(level=0):
        ok = g[g.low_before_high & (g.range > MIN_RANGE)]
        if len(ok) == 0:
            days.append({"date": date, "traded": False, "net": 0.0})
            continue
        best = ok.iloc[ok.range.to_numpy().argmax()]
        days.append({
            "date": date, "traded": True, "stock": best.stock,
            "gross": best.range,
            "net": (1 + best.range) * (1 - SELL_TAX) - 1,
        })
    res = pd.DataFrame(days).set_index("date").sort_index()
    res.to_csv(outdir / "daily_oracle.csv")

    tr = res[res.traded]
    equity = (1 + res.net).cumprod()
    years = (res.index[-1] - res.index[0]).days / 365.25
    total = equity.iloc[-1]
    cagr = total ** (1 / years) - 1

    print(f"=== 日线全知者(每天一次,买最低卖最高,含税)===")
    print(f"区间: {res.index[0].date()} → {res.index[-1].date()} ({years:.2f} 年)")
    print(f"总天数 {len(res)},操作天数 {len(tr)} ({len(tr)/len(res):.1%})")
    print(f"每笔平均毛振幅 {tr.gross.mean()*1e4:.0f}bp,平均净收益 {tr.net.mean()*1e4:.0f}bp")
    print(f"最大单日净收益 {tr.net.max():.2%},最小 {tr.net.min():.2%}")
    print(f"\n3 年总倍数: {total:,.1f}x")
    print(f"年化: {cagr*100:,.0f}%")

    print("\n=== 分年 ===")
    for y, g in res.groupby(res.index.year):
        t = (1 + g.net).prod()
        print(f"  {y}: {(t-1)*100:,.0f}%  (操作 {g.traded.sum()}/{len(g)} 天)")

    print("\n=== 对照 ===")
    print("分钟级全知者(可无限次交易): 单股中位 ~994%/年,跨股换股 ~5.8×10^6%/年")
    print("现实可达(无可知论,z-score 轮动): 25%-35%/年")
    print(f"输出: {outdir / 'daily_oracle.csv'}")


if __name__ == "__main__":
    main()
