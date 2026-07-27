"""06_rotation.py — 共享资金池的跨股轮动回测(组合级,含 0.1% 卖出税)。

与 03_backtest.py(单标的独立满仓)不同,这里模拟真实资金管理:
一个资金池,每天按信号强弱把资金轮入"回归空间最大"的标的,
信号消失/止盈/超时后轮出。决策在 T 日收盘做出,T+1 日收盘成交(无未来函数)。

策略:
  dip        距 30 日高点跌幅最深者优先,跌 ≥x 入场,涨 y 止盈,T 天超时
  zscore     z_30 最低者优先,z < -k 入场,z ≥ 0 退出,T 天超时
  zflow      zscore + 资金流过滤(dshares_5d > 0 才买)
  reversal   固定持有过去 20 日收益最低的 K 只,每 5 日再平衡(纯因子收割)

输出 analysis/output/rotation/results.csv + equity_{name}.csv

用法: .venv/bin/python analysis/06_rotation.py
"""

import itertools

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

W_DIP, W_Z, W_MOM, W_FLOW = 30, 30, 20, 5
INIT_CAPITAL = 1.0


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """日线收盘价与股本面板:行=日期,列=标的。"""
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes, shares = {}, {}
    for sym in stocks:
        df = load_stock(sym)
        closes[sym] = resample_close(df, "1D")
        shares[sym] = df["total_shares"].resample("1D").last().reindex(closes[sym].index)
    return pd.DataFrame(closes).ffill(), pd.DataFrame(shares).ffill()


def build_signals(px: pd.DataFrame, sh: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "dd": px / px.rolling(W_DIP).max() - 1,          # ≤0,越深越优先
        "z": (px - px.rolling(W_Z).mean()) / px.rolling(W_Z).std(),
        "mom": px.pct_change(W_MOM),
        "flow": sh.pct_change(W_FLOW),
    }


def simulate(px: pd.DataFrame, sig: dict[str, pd.DataFrame],
             strategy: str, params: dict) -> tuple[pd.Series, list[dict]]:
    """事件驱动模拟。返回 (日净值序列, 逐笔交易)。"""
    dates = px.index
    cols = px.columns
    P = params["P"]                       # 最大同时持仓数
    cash = INIT_CAPITAL
    pos: dict[str, dict] = {}             # sym -> {shares, entry_px, entry_i}
    pend_buy: list[str] = []              # T 日决策,T+1 执行的队列
    pend_sell: list[str] = []
    trades = []
    equity = []

    start_i = max(W_DIP, W_Z, W_MOM) + 2
    for i in range(start_i, len(dates)):
        today = px.iloc[i]

        # ── 1. 执行昨日队列(T+1 成交)──
        for sym in pend_sell:
            if sym in pos:
                proceeds = pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                cash += proceeds
                trades.append({
                    "stock": sym,
                    "entry_date": dates[pos[sym]["entry_i"]], "exit_date": dates[i],
                    "net_ret": today[sym] * (1 - SELL_TAX) / pos[sym]["entry_px"] - 1,
                    "hold_days": i - pos[sym]["entry_i"],
                })
                del pos[sym]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                alloc = cash / (n_slots - j)      # 剩余现金均分到剩余名额
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i}
                cash -= alloc
        pend_sell, pend_buy = [], []

        # ── 2. 记录净值 ──
        eq = cash + sum(p["shares"] * today[s] for s, p in pos.items())
        equity.append(eq)

        # ── 3. 收盘决策(明日执行)──
        held = list(pos.keys())
        if strategy == "reversal":
            K = params["K"]
            if (i - start_i) % 5 == 0:                 # 每 5 日再平衡
                mom_t = sig["mom"].iloc[i].dropna()
                target = list(mom_t.nsmallest(K).index)
                pend_sell = [s for s in held if s not in target]
                pend_buy = [s for s in target if s not in pos]
        else:
            # 退出规则
            for sym in held:
                p = pos[sym]
                held_days = i - p["entry_i"]
                if strategy == "dip":
                    exit_now = today[sym] >= p["entry_px"] * (1 + params["y"])
                else:  # zscore / zflow
                    exit_now = sig["z"].iloc[i][sym] >= 0
                if exit_now or held_days >= params["T"]:
                    pend_sell.append(sym)
            # 入场规则(信号最強者优先)
            slots = P - len(pos) + len(pend_sell)
            if slots > 0:
                if strategy == "dip":
                    cand = sig["dd"].iloc[i].dropna()
                    cand = cand[(cand <= -params["x"]) & ~cand.index.isin(pos)]
                    ranked = cand.sort_values()         # 最深跌幅优先
                else:
                    cand = sig["z"].iloc[i].dropna()
                    cand = cand[(cand < -params["k"]) & ~cand.index.isin(pos)]
                    if strategy == "zflow":
                        fl = sig["flow"].iloc[i]
                        cand = cand[fl.reindex(cand.index) > 0]
                    ranked = cand.sort_values()         # z 最低优先
                pend_buy = list(ranked.index[:slots])

    eq = pd.Series(equity, index=dates[start_i:])
    return eq, trades


def metrics(eq: pd.Series, trades: list[dict], label: str) -> dict:
    td = pd.DataFrame(trades)
    out = {"config": label, "cagr": cagr(eq), "max_dd": max_drawdown(eq),
           "sharpe": ann_sharpe(eq.pct_change().dropna(), 365.25),
           "n_trades": len(td)}
    if len(td):
        out.update({
            "win_rate": (td.net_ret > 0).mean(),
            "avg_net_bp": td.net_ret.mean() * 1e4,
            "avg_hold_d": td.hold_days.mean(),
        })
    return out


def split_metrics(eq: pd.Series, trades: list[dict], label: str) -> list[dict]:
    rows = []
    spans = {"train": (eq.index[0], SPLIT_TRAIN_END),
             "valid": (SPLIT_TRAIN_END, SPLIT_VALID_END),
             "test": (SPLIT_VALID_END, eq.index[-1])}
    td = pd.DataFrame(trades)
    for name, (a, b) in spans.items():
        sub = eq[(eq.index >= a) & (eq.index < b)]
        if len(sub) < 30:
            continue
        t = td[(td.exit_date >= a) & (td.exit_date < b)] if len(td) else td
        m = metrics(sub / sub.iloc[0], t.to_dict("records"), label)
        m["split"] = name
        rows.append(m)
    return rows


def main() -> None:
    outdir = ensure_out("rotation")
    px, sh = load_panel()
    sig = build_signals(px, sh)

    # 基准:等权买入持有(期末一次性计税)
    bh = (px / px.iloc[0]).mean(axis=1)
    base_rows = split_metrics(bh, [], "equal_weight_buyhold")

    jobs = []
    for x, y, P in itertools.product([0.01, 0.02], [0.01, 0.02], [1, 3, 5]):
        jobs.append(("dip", {"x": x, "y": y, "T": 90, "P": P}))
    for k, P in itertools.product([1.0, 1.5], [1, 3, 5]):
        jobs.append(("zscore", {"k": k, "T": 90, "P": P}))
        jobs.append(("zflow", {"k": k, "T": 90, "P": P}))
    for K in [3, 5]:
        jobs.append(("reversal", {"K": K, "P": K}))

    rows = []
    for strategy, params in jobs:
        label = f"{strategy}_" + "_".join(f"{k}{v:g}" for k, v in params.items())
        eq, trades = simulate(px, sig, strategy, params)
        eq.to_csv(outdir / f"equity_{label}.csv")
        rows.extend(split_metrics(eq, trades, label))
        full = metrics(eq, trades, label)
        full["split"] = "full"
        rows.append(full)
        print(f"完成 {label}", end="\r")

    res = pd.DataFrame(base_rows + rows)
    res.to_csv(outdir / "results.csv", index=False)

    # ── 汇总打印:各配置 full 期 CAGR,及其 train/test 对照 ──
    full = res[res.split == "full"].set_index("config")
    test = res[res.split == "test"].set_index("config")
    valid = res[res.split == "valid"].set_index("config")
    train = res[res.split == "train"].set_index("config")
    view = full[["cagr", "max_dd", "sharpe", "n_trades", "win_rate",
                 "avg_net_bp", "avg_hold_d"]].copy()
    view["cagr_train"] = train["cagr"]
    view["cagr_valid"] = valid["cagr"]
    view["cagr_test"] = test["cagr"]
    view = view.sort_values("cagr", ascending=False)
    pd.set_option("display.width", 250)
    print("\n\n=== 轮动组合回测(按 full 期 CAGR 排序)===")
    print(view.round(4).to_string())
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
