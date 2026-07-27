"""03_backtest.py — 按 Torn 真实交易规则的回测框架 + 策略库。

规则:日线收盘信号,次日收盘成交;买入免税,卖出价 × (1-0.1%);无限量,无滑点。
单标的满仓进出(flat 或 all-in),另汇总等权组合视角。

样本划分:train 2023-07→2025-07 | valid 2025-07→2026-01 | test 2026-01→2026-07。
所有策略结论以样本外(valid+test)表现为准。

输出 analysis/output/backtest/:
  trades.csv       逐笔交易记录
  per_stock.csv    策略×参数×标的×样本段 指标
  summary.csv      策略×参数 在全市场上的汇总(按样本段)
  portfolio.csv    等权组合净值指标(策略×参数×样本段)

用法: .venv/bin/python analysis/03_backtest.py
"""

import itertools

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    cagr, ensure_out, list_stocks, load_stock, max_drawdown, resample_close,
)

# ── 参数网格 ─────────────────────────────────────────────
DIP_GRID = list(itertools.product(
    [30, 60, 120],        # W: 回撤参考窗口(天)
    [0.01, 0.02, 0.03, 0.05],  # x: 买入阈值(距窗口最高点跌幅)
    [0.01, 0.02, 0.03, 0.05],  # y: 止盈
    [30, 90],             # T: 超时退出(天)
))
ZSCORE_GRID = list(itertools.product(
    [30, 60],             # N: 均线窗口(天)
    [1.0, 1.5, 2.0],      # k: 入场 z 阈值(买入 z < -k)
    [30, 90],             # T: 超时退出(天)
))
MOM_GRID = list(itertools.product(
    [20, 60],             # 动量参考窗口(天)
    [30, 90],             # T: 持有上限(天)
))


def run_dip(close: pd.Series, W: int, x: float, y: float, T: int) -> list[dict]:
    """抄底反弹:收盘价距过去 W 天最高点跌 ≥x 时,次日收盘买入;
    涨 y 止盈,或持仓 T 天超时退出(次日收盘卖)。"""
    px = close.values
    idx = close.index
    roll_max = close.rolling(W, min_periods=1).max().values
    trades = []
    i = W  # 需要满窗口
    n = len(px)
    while i < n - 1:
        if px[i] <= roll_max[i] * (1 - x):
            entry_i = i + 1          # 次日收盘成交
            entry_px = px[entry_i]
            exit_i = None
            exit_reason = "timeout"
            for j in range(entry_i + 1, min(entry_i + T + 1, n)):
                if px[j] >= entry_px * (1 + y):
                    exit_i = j + 1 if j + 1 < n else j  # 信号次日卖出
                    exit_reason = "take_profit"
                    break
            if exit_i is None:
                exit_i = min(entry_i + T, n - 1)
            exit_px = px[exit_i]
            trades.append({
                "entry_date": idx[entry_i], "exit_date": idx[exit_i],
                "entry_px": entry_px, "exit_px": exit_px,
                "gross_ret": exit_px / entry_px - 1,
                "net_ret": exit_px * (1 - SELL_TAX) / entry_px - 1,
                "hold_days": (idx[exit_i] - idx[entry_i]).days,
                "reason": exit_reason,
            })
            i = exit_i                # 平仓后才允许再次入场
        else:
            i += 1
    return trades


def run_zscore(close: pd.Series, N: int, k: float, T: int) -> list[dict]:
    """z-score 均值回复:z=(p-MA_N)/std_N < -k 时次日买入;
    z 回到 ≥0 次日卖出,或持仓 T 天超时退出。"""
    ma = close.rolling(N).mean()
    sd = close.rolling(N).std()
    z = ((close - ma) / sd).values
    px = close.values
    idx = close.index
    trades = []
    i = N
    n = len(px)
    while i < n - 1:
        if z[i] < -k:
            entry_i = i + 1
            entry_px = px[entry_i]
            exit_i = None
            exit_reason = "timeout"
            for j in range(entry_i + 1, min(entry_i + T + 1, n)):
                if z[j] >= 0:
                    exit_i = j + 1 if j + 1 < n else j
                    exit_reason = "reverted"
                    break
            if exit_i is None:
                exit_i = min(entry_i + T, n - 1)
            exit_px = px[exit_i]
            trades.append({
                "entry_date": idx[entry_i], "exit_date": idx[exit_i],
                "entry_px": entry_px, "exit_px": exit_px,
                "gross_ret": exit_px / entry_px - 1,
                "net_ret": exit_px * (1 - SELL_TAX) / entry_px - 1,
                "hold_days": (idx[exit_i] - idx[entry_i]).days,
                "reason": exit_reason,
            })
            i = exit_i
        else:
            i += 1
    return trades


def run_momentum(close: pd.Series, W: int, T: int) -> list[dict]:
    """动量对照组:过去 W 天涨幅为正则次日买入,持有 T 天或动量转负卖出。"""
    mom = close.pct_change(W).values
    px = close.values
    idx = close.index
    trades = []
    i = W
    n = len(px)
    while i < n - 1:
        if mom[i] > 0:
            entry_i = i + 1
            entry_px = px[entry_i]
            exit_i = None
            for j in range(entry_i + 1, min(entry_i + T + 1, n)):
                if mom[j] <= 0:
                    exit_i = j + 1 if j + 1 < n else j
                    break
            if exit_i is None:
                exit_i = min(entry_i + T, n - 1)
            exit_px = px[exit_i]
            trades.append({
                "entry_date": idx[entry_i], "exit_date": idx[exit_i],
                "entry_px": entry_px, "exit_px": exit_px,
                "gross_ret": exit_px / entry_px - 1,
                "net_ret": exit_px * (1 - SELL_TAX) / entry_px - 1,
                "hold_days": (idx[exit_i] - idx[entry_i]).days,
                "reason": "signal" if exit_i else "timeout",
            })
            i = exit_i
        else:
            i += 1
    return trades


def split_of(ts: pd.Timestamp) -> str:
    if ts < SPLIT_TRAIN_END:
        return "train"
    if ts < SPLIT_VALID_END:
        return "valid"
    return "test"


def trades_metrics(trades: list[dict], span_days: int) -> dict:
    """由逐笔交易计算资金口径指标(单标的满仓,闲置资金收益 0)。"""
    if not trades:
        return {"n_trades": 0, "cagr": 0.0, "win_rate": np.nan,
                "avg_net_bp": np.nan, "time_in_mkt": 0.0, "tax_paid_pct": 0.0}
    df = pd.DataFrame(trades)
    total_ret = (1 + df["net_ret"]).prod() - 1          # 复利(满仓滚动)
    years = span_days / 365.25
    strat_cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    time_in = df["hold_days"].sum() / span_days
    tax_paid = (df["exit_px"] * SELL_TAX / df["entry_px"]).sum()  # 相对初始资金
    return {
        "n_trades": len(df),
        "cagr": strat_cagr,
        "win_rate": (df["net_ret"] > 0).mean(),
        "avg_net_bp": df["net_ret"].mean() * 1e4,
        "avg_hold_days": df["hold_days"].mean(),
        "time_in_mkt": time_in,
        "tax_paid_pct": tax_paid / max(len(df), 1) * 100,
    }


def bh_metrics(close: pd.Series) -> dict:
    """买入持有(期末一次性卖出,扣卖出税)。"""
    eq = close / close.iloc[0]
    return {"cagr": (1 + cagr(eq)) * (1 - SELL_TAX) - 1,
            "max_drawdown": max_drawdown(eq)}


def main() -> None:
    outdir = ensure_out("backtest")
    stocks = [s for s in list_stocks() if s != "TCSE"]

    all_trades, per_stock_rows = [], []
    closes = {}
    for sym in stocks:
        closes[sym] = resample_close(load_stock(sym), "1D")

    span = {s: (closes[s].index[-1] - closes[s].index[0]).days for s in stocks}

    # ── buy&hold 基准 ──
    for sym in stocks:
        m = bh_metrics(closes[sym])
        per_stock_rows.append({"strategy": "buyhold", "params": "-",
                               "stock": sym, "split": "full", **m})

    # ── 参数策略 ──
    jobs = ([("dip", p, run_dip) for p in DIP_GRID] +
            [("zscore", p, run_zscore) for p in ZSCORE_GRID] +
            [("momentum", p, run_momentum) for p in MOM_GRID])

    for name, params, func in jobs:
        pstr = "_".join(f"{v:g}" for v in params)
        for sym in stocks:
            trades = func(closes[sym], *params)
            for t in trades:
                t.update({"strategy": name, "params": pstr, "stock": sym,
                          "split": split_of(t["entry_date"])})
            all_trades.extend(trades)
            m = trades_metrics(trades, span[sym])
            per_stock_rows.append({"strategy": name, "params": pstr,
                                   "stock": sym, "split": "full", **m})
        print(f"完成 {name} {pstr}", end="\r")

    trades_df = pd.DataFrame(all_trades)
    per_stock = pd.DataFrame(per_stock_rows)
    trades_df.to_csv(outdir / "trades.csv", index=False)
    per_stock.to_csv(outdir / "per_stock.csv", index=False)

    # ── 汇总:策略×参数×样本段(把各标的交易拼起来,按 1/35 资金加权)──
    summary_rows = []
    n_stocks = len(stocks)
    for (name, pstr, split), g in trades_df.groupby(["strategy", "params", "split"]):
        w = g["net_ret"] / n_stocks          # 等权资金贡献
        summary_rows.append({
            "strategy": name, "params": pstr, "split": split,
            "n_trades": len(g),
            "win_rate": (g["net_ret"] > 0).mean(),
            "avg_net_bp": g["net_ret"].mean() * 1e4,
            "avg_gross_bp": g["gross_ret"].mean() * 1e4,
            "avg_hold_days": g["hold_days"].mean(),
            "portfolio_ret_contrib_bp": w.sum() * 1e4,   # 对组合总收益的贡献
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    # ── 打印:每个策略 train 段最优参数,及其样本外表现 ──
    print("\n\n=== 各策略 train 段 Top5 参数(按组合收益贡献)===")
    for name in ["dip", "zscore", "momentum"]:
        sub = summary[(summary.strategy == name)]
        tr = sub[sub.split == "train"].nlargest(
            5, "portfolio_ret_contrib_bp")
        print(f"\n--- {name} (train top5) ---")
        cols = ["params", "n_trades", "win_rate", "avg_net_bp",
                "avg_hold_days", "portfolio_ret_contrib_bp"]
        print(tr[cols].round(2).to_string(index=False))
        # 同参数的样本外
        for p in tr["params"].head(3):
            oos = sub[(sub.params == p) & (sub.split.isin(["valid", "test"]))]
            print(f"  {p} 样本外:")
            print(oos[["split", "n_trades", "win_rate", "avg_net_bp",
                       "portfolio_ret_contrib_bp"]].round(2).to_string(index=False))

    print("\n=== buy&hold 基准(全期,含卖出税)===")
    bh = per_stock[per_stock.strategy == "buyhold"]
    print(f"35 股 CAGR: mean={bh.cagr.mean()*100:.2f}%  "
          f"median={bh.cagr.median()*100:.2f}%  "
          f"maxDD_mean={bh.max_drawdown.mean()*100:.1f}%")

    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
