"""02_demand.py — total_shares 变动(全服玩家净买入代理)的预测力检验。

检验两个方向:
  A. 资金 → 价格: 当期 Δshares% 与未来 1h/4h/1d 收益的相关(玩家买入能否预测/推动价格)
  B. 价格 → 资金: 当期收益与未来 1h/4h/1d Δshares% 的相关(玩家追涨还是抄底)
以及极端资金流分位数的前瞻收益。

输出 analysis/output/demand/:
  ic_flow_to_price.csv   每标的: Δshares → 未来收益的 Pearson/Spearman IC
  ic_price_to_flow.csv   每标的: 收益 → 未来 Δshares 的 IC
  quantile_forward.csv   按 Δshares 十分位的前瞻平均收益(bp)

用法: .venv/bin/python analysis/02_demand.py
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import ensure_out, list_stocks, load_stock, resample_close

BAR = "1h"                       # 基础周期
FWD_HOURS = {"1h": 1, "4h": 4, "1d": 24}   # 前瞻窗口(以 bar 数计)


def flow_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """1h bar 上的:收益、Δshares%、前瞻收益、前瞻 Δshares%。"""
    close = resample_close(df, BAR)
    shares = df["total_shares"].resample(BAR).last().reindex(close.index)
    out = pd.DataFrame({"close": close, "shares": shares}).dropna()
    out["ret"] = out["close"].pct_change()
    out["dshares"] = out["shares"].pct_change()
    for name, h in FWD_HOURS.items():
        out[f"fwd_ret_{name}"] = out["close"].shift(-h) / out["close"] - 1
        out[f"fwd_dshares_{name}"] = out["shares"].shift(-h) / out["shares"] - 1
    return out.dropna()


def ic_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Pearson r、Spearman rho、Spearman p 值。"""
    pear = x.corr(y)
    rho, p = sps.spearmanr(x, y)
    return float(pear), float(rho), float(p)


def main() -> None:
    outdir = ensure_out("demand")
    stocks = [s for s in list_stocks() if s != "TCSE"]

    a_rows, b_rows = {}, {}
    frames = {}
    for sym in stocks:
        f = flow_price_frame(load_stock(sym))
        frames[sym] = f
        a_rows[sym] = {}
        b_rows[sym] = {}
        for name in FWD_HOURS:
            pear, rho, p = ic_pair(f["dshares"], f[f"fwd_ret_{name}"])
            a_rows[sym][f"{name}_pearson"] = pear
            a_rows[sym][f"{name}_spearman"] = rho
            a_rows[sym][f"{name}_p"] = p
            pear, rho, p = ic_pair(f["ret"], f[f"fwd_dshares_{name}"])
            b_rows[sym][f"{name}_pearson"] = pear
            b_rows[sym][f"{name}_spearman"] = rho
            b_rows[sym][f"{name}_p"] = p

    a = pd.DataFrame(a_rows).T
    b = pd.DataFrame(b_rows).T
    a.to_csv(outdir / "ic_flow_to_price.csv")
    b.to_csv(outdir / "ic_price_to_flow.csv")

    print("=== A. Δshares → 未来收益(35 股 Spearman IC 汇总)===")
    for name in FWD_HOURS:
        col = a[f"{name}_spearman"]
        sig = (a[f"{name}_p"] < 0.01).sum()
        print(f"  {name}: mean={col.mean():+.4f}  median={col.median():+.4f}  "
              f"p<0.01 的标的数={sig}/{len(a)}")

    print("\n=== B. 收益 → 未来 Δshares(玩家行为方向)===")
    for name in FWD_HOURS:
        col = b[f"{name}_spearman"]
        sig = (b[f"{name}_p"] < 0.01).sum()
        print(f"  {name}: mean={col.mean():+.4f}  median={col.median():+.4f}  "
              f"p<0.01 的标的数={sig}/{len(b)}")

    # ── 分位数分析:全市场拼接(按标的内十分位,去量纲)──
    q_rows = {}
    for name, h in FWD_HOURS.items():
        buckets = []
        for sym, f in frames.items():
            try:
                q = pd.qcut(f["dshares"], 10, labels=False, duplicates="drop")
            except ValueError:
                continue
            buckets.append(f.groupby(q)[f"fwd_ret_{name}"].mean())
        qdf = pd.concat(buckets, axis=1).mean(axis=1) * 1e4  # bp
        q_rows[name] = qdf
    qt = pd.DataFrame(q_rows)
    qt.index.name = "dshares_decile(0=净卖出最多)"
    qt.to_csv(outdir / "quantile_forward.csv")
    print("\n=== 按 Δshares 十分位的前瞻平均收益(bp,全市场)===")
    print(qt.round(3).to_string())

    # ── Δshares 的典型量级(感受信号强度)──
    mag = pd.Series({s: f["dshares"].abs().median() for s, f in frames.items()})
    print(f"\n|Δshares%|/h 中位数(各股): median={mag.median()*1e4:.2f}bp, "
          f"max={mag.max()*1e4:.2f}bp")

    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
