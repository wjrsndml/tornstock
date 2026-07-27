"""11_predictability.py — 分钟级波动"不可预测性"统计检验成套方案。

逻辑:统计不能证明"不可预测",但能:
  1. 给可预测性定量上界(方差解释率);
  2. 用多类检验搜索残余结构(线性/非线性/符号/波动聚集);
  3. 把残余结构换算成钱,与 0.1% 税线比较(经济上是否可用)。

检验清单(对分钟收益率 r_t):
  A. Ljung-Box:原始序列的自相关是否联合为零(预期:拒绝,已知有 AR(1) 成分)
  B. AR(1) 残差的 Ljung-Box:去掉已知的均值回复成分后是否还有线性结构
  C. 游程检验:涨跌符号序列是否随机
  D. 距离相关 dCor:滞后收益与未来收益的非线性依赖(子样本)
  E. ARCH-LM:波动率是否聚集(可预测的是"波动"而非"方向")
  F. 经济换算:最优 AR(1) 预测每笔期望收益 vs 10bp 税线

用法: .venv/bin/python analysis/11_predictability.py [股票代码...]
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import SELL_TAX, ensure_out, list_stocks, load_stock


def ljung_box(r: np.ndarray, lags: int) -> tuple[float, float]:
    n = len(r)
    r = r - r.mean()
    denom = (r ** 2).sum()
    q = 0.0
    for k in range(1, lags + 1):
        rk = (r[k:] * r[:-k]).sum() / denom
        q += rk ** 2 / (n - k)
    q *= n * (n + 2)
    return q, float(sps.chi2.sf(q, lags))


def runs_test(signs: np.ndarray) -> tuple[float, float]:
    s = signs[signs != 0]
    n1, n2 = (s > 0).sum(), (s < 0).sum()
    runs = 1 + (s[1:] * s[:-1] < 0).sum()
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = max((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
               / ((n1 + n2) ** 2 * (n1 + n2 - 1))), 1e-9)
    z = (runs - mu) / np.sqrt(var)
    return z, float(2 * sps.norm.sf(abs(z)))


def dcor(x: np.ndarray, y: np.ndarray) -> float:
    """距离相关(子样本,O(n^2))。0 = 独立(含非线性)。"""
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    return float(np.sqrt(max(dcov2, 0) / np.sqrt(dvar_x * dvar_y)))


def arch_lm(r: np.ndarray, lags: int = 5) -> tuple[float, float]:
    r2 = r ** 2
    n = len(r2) - lags
    X = np.column_stack([r2[lags - k - 1: n + lags - k - 1]
                         for k in range(lags)])
    X = np.column_stack([np.ones(n), X])
    y = r2[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2_lm = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lm = n * r2_lm
    return lm, float(sps.chi2.sf(lm, lags))


def main() -> None:
    outdir = ensure_out("predictability")
    stocks = sys.argv[1:] or ["SYM", "ASS", "FHG"]
    rows = []

    for sym in stocks:
        px = load_stock(sym)["price"].to_numpy()
        r = np.diff(np.log(px))                     # 分钟对数收益
        r = r[np.isfinite(r)]
        n = len(r)

        # A. 原始序列 Ljung-Box
        q10, p10 = ljung_box(r, 10)

        # B. AR(1) 拟合后的残差
        rho1 = np.corrcoef(r[1:], r[:-1])[0, 1]
        resid = r[1:] - rho1 * r[:-1]
        qr, pr = ljung_box(resid, 10)

        # C. 游程检验
        zrun, prun = runs_test(np.sign(resid))

        # D. 距离相关(子样本防 O(n^2) 爆炸)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(resid) - 1, 4000, replace=False)
        dc = dcor(r[idx], r[idx + 1])              # 原始:应≈|rho1|(线性依赖)
        dc_res = dcor(resid[idx], resid[idx + 1])  # 残差:非线性残余依赖

        # E. ARCH 效应
        lm, parch = arch_lm(r)

        # F. 经济换算:AR(1) 最优预测的期望幅度 vs 税
        sigma_min = r.std()
        exp_edge_bp = abs(rho1) * sigma_min * np.sqrt(2 / np.pi) * 1e4
        r2_pct = rho1 ** 2 * 100

        rows.append({
            "stock": sym, "lag1自相关": rho1,
            "方向可解释方差%": r2_pct,
            "LB原始p值": p10, "LB残差p值": pr,
            "游程p值": prun, "dCor原始": dc, "dCor残差": dc_res,
            "ARCH_p值": parch,
            "AR1最优预测每笔期望bp": exp_edge_bp,
            "税线bp": SELL_TAX * 1e4,
            "差距倍数": SELL_TAX * 1e4 / exp_edge_bp,
        })

    res = pd.DataFrame(rows).set_index("stock")
    res.to_csv(outdir / "predictability_tests.csv")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string())
    print(f"""
读法(注意:145 万样本下任何微观结构 p 值都为 0,要看效应量):
  方向可解释方差% → 下一分钟方向最多有 ~0.3-0.4% 的方差可被预测
  LB残差p值=0 但各 lag 自相关<1.5% → 残余线性结构存在但量级可忽略
  dCor残差>噪声底(~0.016) → 残差有非线性依赖,但结合 ARCH_p值≈0
    可知它是"波动聚集"(能预测抖动大小),不是"方向"(预测不了涨跌)
  差距倍数 → 最优方向预测每笔期望收益比 0.1% 税线小 ~160-185 倍
""")
    print(f"输出: {outdir / 'predictability_tests.csv'}")


if __name__ == "__main__":
    main()
