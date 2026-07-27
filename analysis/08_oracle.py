"""08_oracle.py — 全知者(完美择时)收益上限,动态规划精确求解。

规则与游戏一致:任意时刻可按当分钟报价买入/卖出(无限量),
买入免税,卖出 -0.1%。全知者知道全部未来价格,求复利最大化的交易序列。

DP 定义(分钟级):
  F  = 当前空仓时的最大资金倍数
  Q  = 当前持仓时的最大股数(每股初始 1 元口径)
  每分钟转移:F ← max(F, Q·p·(1-tax));  Q ← max(Q, F/p)

  单股:  逐只求解。
  跨股:  F 为标量,Q[s] 每只一份 —— 等价于"同一时刻只持一股、可随时换股"
         的资金池上限(卖 s 买 s' 允许同一分钟完成)。

输出 analysis/output/oracle/oracle_results.csv + 终端汇总。

用法: .venv/bin/python analysis/08_oracle.py
"""

import time

import numpy as np
import pandas as pd

from common import SELL_TAX, ensure_out, list_stocks, load_stock

YEARS = 3.0  # 数据跨度


def oracle_single(px: np.ndarray) -> tuple[float, int]:
    """单股全知者。返回 (最终资金倍数, 完成回合数)。"""
    F = 1.0          # 空仓资金倍数
    Q = 0.0          # 持仓股数
    nF = 0           # F 路径上已完成的回合数
    nQ = 0           # Q 路径上的回合数
    keep = 1.0 - SELL_TAX
    for p in px:
        # 卖出
        f_sell = Q * p * keep
        if f_sell > F:
            F = f_sell
            nF = nQ + 1
        # 买入(允许同一分钟卖后买)
        q_buy = F / p
        if q_buy > Q:
            Q = q_buy
            nQ = nF
    final = max(F, Q * px[-1] * keep)
    return final, max(nF, nQ)


def oracle_multi(panel: np.ndarray) -> float:
    """跨股全知者(同时只持一股,可随时换股)。panel: (n_minutes, n_stocks)。"""
    n, n_stocks = panel.shape
    F = 1.0
    Q = np.zeros(n_stocks)
    keep = 1.0 - SELL_TAX
    for i in range(n):
        row = panel[i]
        # 卖出最优的一只
        f_sell = (Q * row).max() * keep
        if f_sell > F:
            F = f_sell
        # 用最新资金买入(允许同一分钟换股)
        q_buy = F / row
        np.maximum(Q, q_buy, out=Q)
    final = max(F, float((Q * panel[-1]).max()) * keep)
    return final


def main() -> None:
    outdir = ensure_out("oracle")
    stocks = [s for s in list_stocks() if s != "TCSE"]

    # ── 单股全知者 ──
    rows, panels = [], []
    ref_index = None
    for sym in stocks:
        df = load_stock(sym)
        px = df["price"].to_numpy()
        t0 = time.time()
        final, n_trades = oracle_single(px)
        rows.append({
            "stock": sym,
            "oracle_total_ret": final - 1,
            "oracle_cagr": final ** (1 / YEARS) - 1,
            "round_trips": n_trades,
            "trades_per_year": n_trades / YEARS,
        })
        print(f"[{sym}] 全知者 {final:,.1f}x, {n_trades} 回合 "
              f"({time.time()-t0:.0f}s)", flush=True)
        s = df["price"].rename(sym)
        panels.append(s)
        ref_index = df.index if ref_index is None else ref_index

    res = pd.DataFrame(rows).set_index("stock").sort_values(
        "oracle_cagr", ascending=False)
    res.to_csv(outdir / "oracle_results.csv")

    print("\n=== 单股全知者(分钟级,含税,3 年复利)===")
    show = res.assign(
        total=lambda d: (d.oracle_total_ret * 100).round(0),
        cagr=lambda d: (d.oracle_cagr * 100).round(1),
    )[["total", "cagr", "trades_per_year"]]
    show.columns = ["3年总收益%", "年化%", "回合/年"]
    print(show.to_string())
    print(f"\n中位数年化: {res.oracle_cagr.median()*100:.1f}%  "
          f"等权平均年化: {res.oracle_cagr.mean()*100:.1f}%")

    # ── 跨股全知者 ──
    print("\n构建跨股分钟面板...")
    panel_df = pd.concat(panels, axis=1).sort_index().ffill().dropna()
    panel = panel_df.to_numpy(dtype=np.float64)
    print(f"面板: {panel.shape[0]:,} 分钟 × {panel.shape[1]} 股,开始 DP...")
    t0 = time.time()
    final = oracle_multi(panel)
    print(f"\n=== 跨股全知者(同时只持一股,可随时换股)===")
    print(f"3 年总倍数: {final:,.0f}x  →  年化 {(final**(1/YEARS)-1)*100:,.0f}%  "
          f"(耗时 {time.time()-t0:.0f}s)")

    pd.DataFrame([{
        "stock": "MULTI_SWITCH", "oracle_total_ret": final - 1,
        "oracle_cagr": final ** (1 / YEARS) - 1,
        "round_trips": np.nan, "trades_per_year": np.nan,
    }]).set_index("stock").to_csv(outdir / "oracle_multi.csv")

    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
