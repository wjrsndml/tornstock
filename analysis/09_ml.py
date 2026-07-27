"""09_ml.py — 机器学习能否逼近全知者?walk-forward 实证。

方法:
  - 特征:05_factors.py 的全部 16 个日线因子
  - 目标:未来 14 日收益(次日收盘买入口径,与 05/06 一致)
  - 模型:HistGradientBoostingRegressor(35 股合并样本)
  - 验证:walk-forward —— 用过去 400 天训练,预测未来 20 天,滚动推进,
    全程只用当时可得的数据(无未来函数)
  - 对照:单因子 -z_20(已知最强因子)
  - 落地:用 OOS 预测做轮动(预测排名前 3、预测值 > 0 才买,
    z_30 ≥ 0 或 90 天退出),与 06_rotation 的 zscore 策略同规则对照

输出 analysis/output/ml/ml_predictions.csv, ml_strategy.csv

用法: .venv/bin/python analysis/09_ml.py
"""

from importlib import import_module

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingRegressor

from common import (
    SELL_TAX, SPLIT_TRAIN_END, SPLIT_VALID_END,
    cagr, ensure_out, list_stocks, load_stock, max_drawdown, resample_close,
)

f05 = import_module("05_factors")

TRAIN_DAYS = 400
STEP_DAYS = 20
ENTRY_TH = 0.0          # 预测收益 > 0 才买
P = 3                   # 最大同时持仓
T_EXIT = 90             # 超时退出(天)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (特征+目标大表, 收盘价面板, z_30 面板)。"""
    stocks = [s for s in list_stocks() if s != "TCSE"]
    frames, closes = [], {}
    for sym in stocks:
        df = load_stock(sym)
        close = resample_close(df, "1D")
        shares = df["total_shares"].resample("1D").last().reindex(close.index)
        f = f05.compute_factors(close, shares)
        f["stock"] = sym
        frames.append(f)
        closes[sym] = close
    data = pd.concat(frames).dropna()
    px = pd.DataFrame(closes).ffill()
    z30 = (px - px.rolling(30).mean()) / px.rolling(30).std()
    return data, px, z30


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """滚动训练预测,返回 OOS 预测表(date, stock, pred, fwd_14d)。"""
    feat_cols = [c for c in data.columns if not c.startswith(("fwd_", "stock"))]
    dates = np.sort(data.index.get_level_values(0).unique()
                    if isinstance(data.index, pd.MultiIndex) else data.index.unique())
    preds = []
    start = TRAIN_DAYS
    while start < len(dates):
        tr_idx = dates[start - TRAIN_DAYS:start]
        te_idx = dates[start:start + STEP_DAYS]
        tr = data.loc[data.index.isin(tr_idx)]
        te = data.loc[data.index.isin(te_idx)]
        if len(tr) < 1000 or len(te) == 0:
            start += STEP_DAYS
            continue
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=6,
            min_samples_leaf=200, random_state=0)
        model.fit(tr[feat_cols], tr["fwd_14d"])
        out = te[["fwd_14d", "z_20", "stock"]].copy()
        out["pred"] = model.predict(te[feat_cols])
        preds.append(out)
        start += STEP_DAYS
    return pd.concat(preds)


def strategy_sim(pred: pd.DataFrame, px: pd.DataFrame,
                 z30: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """用 OOS 预测做轮动,规则与 06 zscore 策略一致(T+1 成交)。"""
    dates = px.index
    cash, pos, trades, equity = 1.0, {}, [], []
    pend_buy, pend_sell = [], []
    pred_map = {(d, s): v for d, s, v in
                zip(pred.index, pred["stock"], pred["pred"])}

    for i, d in enumerate(dates):
        today = px.iloc[i]
        for sym in pend_sell:
            if sym in pos:
                cash += pos[sym]["shares"] * today[sym] * (1 - SELL_TAX)
                trades.append({"net_ret": today[sym] * (1 - SELL_TAX)
                               / pos[sym]["entry_px"] - 1,
                               "exit_date": d})
                del pos[sym]
        n_slots = min(P - len(pos), len(pend_buy))
        for j, sym in enumerate(pend_buy[:n_slots]):
            if sym not in pos and cash > 1e-12:
                alloc = cash / (n_slots - j)
                pos[sym] = {"shares": alloc / today[sym], "entry_px": today[sym],
                            "entry_i": i}
                cash -= alloc
        pend_buy, pend_sell = [], []
        equity.append(cash + sum(p["shares"] * today[s] for s, p in pos.items()))

        for sym, p in pos.items():
            if z30.iloc[i][sym] >= 0 or i - p["entry_i"] >= T_EXIT:
                pend_sell.append(sym)
        slots = P - len(pos) + len(pend_sell)
        if slots > 0:
            cand = {}
            for sym in px.columns:
                if sym in pos:
                    continue
                v = pred_map.get((d, sym))
                if v is not None and v > ENTRY_TH:
                    cand[sym] = v
            pend_buy = sorted(cand, key=cand.get, reverse=True)[:slots]

    return pd.Series(equity, index=dates), trades


def main() -> None:
    outdir = ensure_out("ml")
    data, px, z30 = build_dataset()
    feat_cols = [c for c in data.columns if not c.startswith(("fwd_", "stock"))]
    print(f"样本: {len(data):,} 行 × {len(feat_cols)} 特征")

    pred = walk_forward(data)
    pred.to_csv(outdir / "ml_predictions.csv")
    print(f"OOS 预测: {len(pred):,} 行 "
          f"({pred.index.min().date()} → {pred.index.max().date()})")

    # ── 预测质量:ML vs 单因子 -z_20 ──
    ic_ml, ic_z = [], []
    for d, g in pred.groupby(level=0):
        if len(g) < 10:
            continue
        ic_ml.append(sps.spearmanr(g["pred"], g["fwd_14d"])[0])
        ic_z.append(sps.spearmanr(-g["z_20"], g["fwd_14d"])[0])
    print("\n=== 样本外 IC(每日截面,14 日前瞻收益)===")
    print(f"ML 模型:  mean={np.nanmean(ic_ml):.4f}  "
          f"t={np.nanmean(ic_ml)/(np.nanstd(ic_ml)/np.sqrt(len(ic_ml))):.1f}")
    print(f"-z_20:    mean={np.nanmean(ic_z):.4f}  "
          f"t={np.nanmean(ic_z)/(np.nanstd(ic_z)/np.sqrt(len(ic_z))):.1f}")

    # ── 策略回测(仅 OOS 段)──
    oos_start = pred.index.min()
    px_oos = px[px.index >= oos_start]
    z_oos = z30.reindex(px_oos.index)
    eq, trades = strategy_sim(pred, px_oos, z_oos)
    td = pd.DataFrame(trades)

    rows = []
    spans = {"OOS全段": (eq.index[0], eq.index[-1]),
             "其中test段": (SPLIT_VALID_END, eq.index[-1])}
    for name, (a, b) in spans.items():
        sub = eq[(eq.index >= a) & (eq.index <= b)]
        t = td[td.exit_date >= a] if len(td) else td
        rows.append({
            "span": name, "cagr": cagr(sub / sub.iloc[0]),
            "max_dd": max_drawdown(sub / sub.iloc[0]),
            "n_trades": len(t),
            "win_rate": (t.net_ret > 0).mean() if len(t) else np.nan,
            "avg_net_bp": t.net_ret.mean() * 1e4 if len(t) else np.nan,
        })
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "ml_strategy.csv", index=False)
    print("\n=== ML 轮动策略(对照:06 zscore 日线 24.3% / 小时级 26.8%)===")
    print(res.round(4).to_string(index=False))
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
