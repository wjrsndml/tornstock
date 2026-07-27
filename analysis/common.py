"""分析脚本共用工具:数据加载、重采样、常量。"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "merged"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SELL_TAX = 0.001  # 卖出税 0.1%,买入免税

# 样本划分(防过拟合):训练 / 验证 / 测试
SPLIT_TRAIN_END = pd.Timestamp("2025-07-23", tz="UTC")
SPLIT_VALID_END = pd.Timestamp("2026-01-23", tz="UTC")
# 测试段: 2026-01-23 → 2026-07-23


def list_stocks() -> list[str]:
    """全部标的代码(含 TCSE 指数)。"""
    return sorted(p.stem for p in DATA_DIR.glob("*.parquet"))


def load_stock(symbol: str) -> pd.DataFrame:
    """加载单只标的,返回以 UTC datetime 为索引的 DataFrame(price, total_shares)。"""
    df = pd.read_parquet(DATA_DIR / f"{symbol}.parquet")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    return df[["price", "total_shares"]]


def resample_close(df: pd.DataFrame, rule: str) -> pd.Series:
    """分钟价重采样为指定周期的收盘价序列(丢弃空周期)。"""
    return df["price"].resample(rule).last().dropna()


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤(负数,如 -0.25 表示 -25%)。"""
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def cagr(equity: pd.Series) -> float:
    """按首尾与时长计算年化收益率。"""
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if days <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (365.25 / days) - 1.0)


def ann_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """年化夏普(无风险利率取 0)。"""
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def ensure_out(subdir: str) -> Path:
    out = OUTPUT_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
