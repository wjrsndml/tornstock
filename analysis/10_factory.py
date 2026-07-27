"""10_factory.py — 简单策略工厂:批量生成并回测"买跌得狠的"类朴素策略。

策略空间(全部组合,日线,T 日信号 T+1 收盘成交,卖出 -0.1% 税):
  信号 signal ∈ {ret(区间收益), dd(距区间高点跌幅), z(均线偏离标准分)}
  窗口 W ∈ {1, 3, 7, 14, 30, 60}
  持仓数 P ∈ {1, 2, 4}
  退出 exit:
    ("profit", y, T)  持仓毛收益 ≥ y 卖出;T 天未达成强制退出(T=0 表示永久持有)
    ("horizon", H)    固定持有 H 天
    ("rotate", R)     每 R 天再平衡:卖出跌出"最差 P 名"的,买入新进入的

用户原始策略对应:
  #1 买入30天跌最狠,涨了就卖,不涨永久持有 → ret_30, P1, profit y>0(用0.1%), T=0
  #2 买昨天跌最狠,次日涨>0.1%就卖         → ret_1,  P1, profit y=0.001, T=1
  #3 4份资金,卖涨最多的买跌最狠            → ret_W,  P4, rotate R=1

输出 analysis/output/factory/factory_results.csv

用法: .venv/bin/python analysis/10_factory.py
"""

import itertools

import numpy as np
import pandas as pd

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    ann_sharpe, cagr, ensure_out, list_stocks, load_stock, max_drawdown,
    resample_close,
)

WS = [1, 3, 7, 14, 30, 60]
PS = [1, 2, 4]
PROFIT_EXITS = [(y, T) for y in [0.001, 0.005, 0.01, 0.02]
                for T in [7, 30, 90, 0]]           # T=0 → 永久持有
HORIZON_EXITS = [1, 3, 7, 14]
ROTATE_EXITS = [1, 3, 7]


def load_panel() -> pd.DataFrame:
    stocks = [s for s in list_stocks() if s != "TCSE"]
    closes = {s: resample_close(load_stock(s), "1D") for s in stocks}
    return pd.DataFrame(closes).ffill()


def build_signals(px: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    sig = {"ret": {}, "dd": {}, "z": {}}
    for W in WS:
        sig["ret"][W] = px.pct_change(W).to_numpy()
        sig["dd"][W] = (px / px.rolling(W).max() - 1).to_numpy()
        ma, sd = px.rolling(W).mean(), px.rolling(W).std()
        sig["z"][W] = ((px - ma) / sd).to_numpy()
    return sig


def simulate(pxv: np.ndarray, dates, s: np.ndarray, P: int,
             exit_rule: tuple) -> tuple[np.ndarray, list[float], list[int]]:
    """返回 (净值数组, 每笔净收益列表, 每笔持仓天数列表)。s[i] 为第 i 天信号行。"""
    n, n_stocks = pxv.shape
    kind = exit_rule[0]
    cash = 1.0
    pos: dict[int, dict] = {}          # col_idx -> {shares, entry_px, entry_i}
    pend_buy: list[int] = []
    pend_sell: list[int] = []
    equity = np.empty(n)
    rets, holds = [], []
    keep = 1.0 - SELL_TAX

    for i in range(n):
        row = pxv[i]
        for c in pend_sell:
            if c in pos:
                cash += pos[c]["shares"] * row[c] * keep
                rets.append(row[c] * keep / pos[c]["entry_px"] - 1)
                holds.append(i - pos[c]["entry_i"])
                del pos[c]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, c in enumerate(pend_buy[:n_slots]):
            if c not in pos and cash > 1e-12 and np.isfinite(row[c]):
                alloc = cash / (n_slots - j)
                pos[c] = {"shares": alloc / row[c], "entry_px": row[c],
                          "entry_i": i}
                cash -= alloc
        pend_buy, pend_sell = [], []
        equity[i] = cash + sum(p["shares"] * row[c] for c, p in pos.items())

        sr = s[i]
        if not np.isfinite(sr).any():
            continue

        if kind == "rotate":
            R = exit_rule[1]
            if i % R != 0:
                continue
            order = np.argsort(sr)     # 信号最小(跌最狠)优先
            target = [c for c in order if np.isfinite(sr[c])][:P]
            pend_sell = [c for c in pos if c not in target]
            pend_buy = [c for c in target if c not in pos]
        else:
            for c, p in list(pos.items()):
                held = i - p["entry_i"]
                if kind == "profit":
                    _, y, T = exit_rule
                    hit = row[c] >= p["entry_px"] * (1 + y)
                    timeout = T > 0 and held >= T
                    if hit or timeout:
                        pend_sell.append(c)
                else:  # horizon
                    if held >= exit_rule[1]:
                        pend_sell.append(c)
            slots = P - len(pos) + len(pend_sell)
            if slots > 0:
                order = np.argsort(sr)
                ranked = [c for c in order
                          if np.isfinite(sr[c]) and c not in pos]
                pend_buy = ranked[:slots]

    return equity, rets, holds


def main() -> None:
    outdir = ensure_out("factory")
    px = load_panel()
    pxv = px.to_numpy()
    dates = px.index
    sigs = build_signals(px)
    warmup = max(WS) + 2

    def metrics(eq: np.ndarray, rets, holds) -> dict:
        eqs = pd.Series(eq[warmup:], index=dates[warmup:])
        td = pd.DataFrame({"net": rets, "hold": holds,
                           "i_exit": np.cumsum(holds) if holds else []})
        out = {}
        spans = {"train": (eqs.index[0], SPLIT_TRAIN_END),
                 "valid": (SPLIT_TRAIN_END, SPLIT_VALID_END),
                 "test": (SPLIT_VALID_END, eqs.index[-1]),
                 "full": (eqs.index[0], eqs.index[-1])}
        for name, (a, b) in spans.items():
            sub = eqs[(eqs.index >= a) & (eqs.index < b)]
            if len(sub) < 30:
                out[f"cagr_{name}"] = np.nan
                continue
            out[f"cagr_{name}"] = cagr(sub / sub.iloc[0])
        out["max_dd"] = max_drawdown(eqs)
        out["sharpe"] = ann_sharpe(eqs.pct_change().dropna(), 365.25)
        out["n_trades"] = len(rets)
        out["win_rate"] = float(np.mean([r > 0 for r in rets])) if rets else np.nan
        out["avg_net_bp"] = float(np.mean(rets)) * 1e4 if rets else np.nan
        out["avg_hold_d"] = float(np.mean(holds)) if holds else np.nan
        return out

    rows = []
    jobs = []
    for sname, W, P in itertools.product(["ret", "dd", "z"], WS, PS):
        for y, T in PROFIT_EXITS:
            jobs.append((sname, W, P, ("profit", y, T)))
        for H in HORIZON_EXITS:
            jobs.append((sname, W, P, ("horizon", H)))
        for R in ROTATE_EXITS:
            jobs.append((sname, W, P, ("rotate", R)))

    print(f"总组合数: {len(jobs)}")
    for sname, W, P, erule in jobs:
        eq, rets, holds = simulate(pxv, dates, sigs[sname][W], P, erule)
        m = metrics(eq, rets, holds)
        m.update({"signal": sname, "W": W, "P": P,
                  "exit": "_".join(str(x) for x in erule)})
        rows.append(m)

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "factory_results.csv", index=False)

    # ── 汇总 ──
    pd.set_option("display.width", 250)
    cols = ["signal", "W", "P", "exit", "cagr_train", "cagr_valid",
            "cagr_test", "cagr_full", "max_dd", "win_rate",
            "n_trades", "avg_net_bp"]

    print("\n=== 用户三个原始策略 ===")
    u1 = res.query("signal=='ret' and W==30 and P==1 and exit=='profit_0.001_0'")
    u2 = res.query("signal=='ret' and W==1 and P==1 and exit=='profit_0.001_1'")
    u3 = res.query("signal=='ret' and P==4 and exit=='rotate_1'")
    print(pd.concat([u1, u2, u3])[cols].round(4).to_string(index=False))

    print("\n=== test CAGR 分布(全部组合)===")
    print(res.cagr_test.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())

    print("\n=== 按 train 选优 Top15,看 test 表现 ===")
    top = res.nlargest(15, "cagr_train")
    print(top[cols].round(4).to_string(index=False))

    print("\n=== 按 test 直接选优 Top10(仅供对照,含运气成分)===")
    top_t = res.nlargest(10, "cagr_test")
    print(top_t[cols].round(4).to_string(index=False))

    corr = res[["cagr_train", "cagr_test"]].corr().iloc[0, 1]
    print(f"\ntrain/test CAGR 相关性: {corr:.3f} "
          f"(越高说明改进可迁移,越低说明调参≈过拟合)")
    print(f"输出目录: {outdir}")


if __name__ == "__main__":
    main()
