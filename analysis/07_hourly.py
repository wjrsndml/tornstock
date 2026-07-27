"""07_hourly.py — 小时级执行的轮动回测:检验更细的成交时点能否提升收益。

06_rotation.py 用日线(信号次日收盘成交);这里用 1h bar(信号下一小时成交),
只跑 06 中表现最好的几个配置,回答"年化能否再上一个台阶"。

输出 analysis/output/rotation/hourly_results.csv

用法: .venv/bin/python analysis/07_hourly.py
"""

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)
from importlib import import_module

rot = import_module("06_rotation")  # 复用 simulate / split_metrics / 信号参数

HOURS = {"W_DIP": 30 * 24, "W_Z": 30 * 24, "W_MOM": 20 * 24, "W_FLOW": 5 * 24}

TOP_CONFIGS = [
    ("zscore", {"k": 1.5, "T": 90 * 24, "P": 3}),
    ("zflow",  {"k": 1.5, "T": 90 * 24, "P": 3}),
    ("zscore", {"k": 1.0, "T": 90 * 24, "P": 5}),
    ("dip",    {"x": 0.01, "y": 0.01, "T": 90 * 24, "P": 1}),
]


def load_panel_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1h")
        shares[sym] = df["total_shares"].resample("1h").last().reindex(closes[sym].index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def main() -> None:
    outdir = ensure_out("rotation")
    px, sh = load_panel_hourly()
    sig = {
        "dd": px / px.rolling(HOURS["W_DIP"]).max() - 1,
        "z": (px - px.rolling(HOURS["W_Z"]).mean()) / px.rolling(HOURS["W_Z"]).std(),
        "mom": px.pct_change(HOURS["W_MOM"]),
        "flow": sh.pct_change(HOURS["W_FLOW"]),
    }
    # simulate 内的 start_i 用的是日频窗口常量,这里直接用大窗口跳过预热段
    warmup = max(HOURS.values()) + 2

    rows = []
    for strategy, params in TOP_CONFIGS:
        label = f"hourly_{strategy}_" + "_".join(f"{k}{v:g}" for k, v in params.items())
        eq, trades = rot.simulate(px, sig, strategy, params)
        eq = eq[eq.index >= eq.index[warmup]]  # 与日线口径对齐预热期
        eq = eq / eq.iloc[0]
        eq_d = eq.resample("1D").last()         # 指标按日频口径计算(Sharpe 可比)
        for m in rot.split_metrics(eq_d, trades, label):
            rows.append(m)
        full = rot.metrics(eq_d, trades, label)
        full["split"] = "full"
        rows.append(full)
        print(f"完成 {label}", end="\r")

    for r in rows:                               # 持仓时长从小时换算回天
        if "avg_hold_d" in r:
            r["avg_hold_d"] = r["avg_hold_d"] / 24

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "hourly_results.csv", index=False)

    full = res[res.split == "full"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate", "avg_net_bp"]].copy()
    view["cagr_train"] = train["cagr"]
    view["cagr_valid"] = valid["cagr"]
    view["cagr_test"] = test["cagr"]
    print("\n\n=== 小时级执行(对照 06 的日线结果)===")
    print(view.sort_values("cagr", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
