#!/usr/bin/env python3
"""
Torn Stock M1 (1分钟精度) 数据抓取脚本
===========================================
- 从 tornsy.com API 抓取最近 3 年、35 只股票的分钟级 OHLC 数据
- 10 并发，约 10 req/s 速率
- 支持断点续传（已完成的分块会自动跳过）
- 输出：每只股票合并为单个 Parquet 文件

数据量预估：
  35 只股票 × 1,576,800 分钟 ÷ 2000 条/次 ≈ 27,615 次请求
  10 并发全速运行预计耗时：45~60 分钟

用法：
  python fetch_m1_data.py                # 抓取全部 35 只股票
  python fetch_m1_data.py --stocks ASS,LSC,TCSE  # 只抓指定股票
  python fetch_m1_data.py --years 2       # 只抓最近 2 年
  python fetch_m1_data.py --resume        # 断点续传（默认开启）
  python fetch_m1_data.py --no-resume     # 从头开始
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm as async_tqdm

# ── 配置 ───────────────────────────────────────────────
API_BASE = "https://tornsy.com/api"
CHUNK_SIZE = 2000          # API 单次返回上限（分钟数据）
CONCURRENCY = 10           # 并发数
RATE_PER_SEC = 10          # 全局速率限制（次/秒）
REQUEST_TIMEOUT = 30       # 单次请求超时（秒）
MAX_RETRIES = 5            # 失败重试次数
RETRY_BASE_DELAY = 1.0     # 重试基础等待（秒）
YEARS = 3                  # 默认抓取年数
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
PROGRESS_FILE = DATA_DIR / "progress.json"
MERGED_DIR = DATA_DIR / "merged"
LOG_FILE = DATA_DIR / "fetch.log"

# ── 日志 ───────────────────────────────────────────────
def setup_logging():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = None  # 在 main 中初始化


# ── 工具函数 ───────────────────────────────────────────
def now_ts() -> int:
    """当前 Unix 时间戳（秒）"""
    return int(time.time())


def ts_to_str(ts: int) -> str:
    """时间戳转可读字符串"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def round_down_to_minute(ts: int) -> int:
    """向下取整到分钟"""
    return ts // 60 * 60


def compute_chunks(start_ts: int, end_ts: int, chunk_minutes: int = CHUNK_SIZE):
    """
    将 [start_ts, end_ts) 按 chunk_minutes 分钟切分为任务列表。
    返回 [(from_ts, to_ts), ...]
    """
    chunk_secs = chunk_minutes * 60
    tasks = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + chunk_secs, end_ts)
        tasks.append((cur, nxt))
        cur = nxt
    return tasks


# ── API 请求 ───────────────────────────────────────────
class RateLimiter:
    """简单的令牌桶限流器"""

    def __init__(self, rate: float):
        self.rate = rate
        self.interval = 1.0 / rate
        self._next_ok = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_ok = time.monotonic() + self.interval


async def fetch_chunk(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    stock: str,
    from_ts: int,
    to_ts: int,
) -> list[dict] | None:
    """
    抓取一只股票一个时间分块的 M1 数据。
    返回 [{"timestamp": int, "price": float, "total_shares": int}, ...]
    失败返回 None。
    """
    url = f"{API_BASE}/{stock.lower()}"
    params = {
        "interval": "m1",
        "from": from_ts,
        "to": to_ts,
        "limit": CHUNK_SIZE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                await rate_limiter.acquire()
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", str(2 ** attempt)))
                        logger.warning(
                            f"[{stock}] 429 被限流，等待 {retry_after}s (第 {attempt} 次重试)"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        logger.warning(
                            f"[{stock}] HTTP {resp.status} (第 {attempt} 次重试): "
                            f"{from_ts} → {to_ts}"
                        )
                        await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                        continue

                    body = await resp.json()
                    raw_data = body.get("data", [])

                    if not raw_data:
                        # 空数据 —— 可能已经超出历史范围
                        return []

                    # 解析：m1 格式 [timestamp, price, total_shares]
                    records = []
                    for row in raw_data:
                        records.append({
                            "timestamp": int(row[0]),
                            "price": float(row[1]),
                            "total_shares": int(row[2]),
                        })
                    return records

        except asyncio.TimeoutError:
            logger.warning(
                f"[{stock}] 请求超时 (第 {attempt} 次重试): {from_ts} → {to_ts}"
            )
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        except aiohttp.ClientError as e:
            logger.warning(f"[{stock}] 网络错误: {e} (第 {attempt} 次重试)")
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        except Exception as e:
            logger.error(f"[{stock}] 未知错误: {e}")
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    logger.error(f"[{stock}] 耗尽重试次数: {from_ts} → {to_ts}")
    return None


# ── 进度管理 ───────────────────────────────────────────
def load_progress() -> dict:
    """加载已完成的 chunk 记录"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_progress(progress: dict):
    """保存进度"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def chunk_key(stock: str, from_ts: int, to_ts: int) -> str:
    return f"{stock}_{from_ts}_{to_ts}"


def chunk_done(progress: dict, stock: str, from_ts: int, to_ts: int) -> bool:
    return chunk_key(stock, from_ts, to_ts) in progress


def mark_chunk_done(progress: dict, stock: str, from_ts: int, to_ts: int):
    progress[chunk_key(stock, from_ts, to_ts)] = True
    # 每 100 个 chunk 持久化一次，避免频繁 IO
    if len(progress) % 100 == 0:
        save_progress(progress)


# ── 股票列表 ───────────────────────────────────────────
async def fetch_stock_list(session: aiohttp.ClientSession) -> list[str]:
    """从 API 获取全部股票列表"""
    url = f"{API_BASE}/stocks"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
        if resp.status != 200:
            logger.error(f"无法获取股票列表: HTTP {resp.status}")
            return []
        body = await resp.json()
        stocks = [s["stock"] for s in body.get("data", [])]
        logger.info(f"获取到 {len(stocks)} 只股票: {', '.join(stocks)}")
        return stocks


# ── 写入 Parquet ───────────────────────────────────────
def write_chunk_parquet(stock: str, from_ts: int, records: list[dict]):
    """将一个 chunk 的数据写入临时 Parquet 文件"""
    if not records:
        return
    stock_dir = CHUNKS_DIR / stock
    stock_dir.mkdir(parents=True, exist_ok=True)
    fpath = stock_dir / f"{from_ts}.parquet"

    df = pd.DataFrame(records)
    df["stock"] = stock
    table = pa.Table.from_pandas(df)
    pq.write_table(table, fpath, compression="zstd")


def merge_stock(stock: str) -> Path | None:
    """合并某只股票所有 chunk → 单个 Parquet 文件"""
    stock_dir = CHUNKS_DIR / stock
    if not stock_dir.exists():
        return None

    files = sorted(stock_dir.glob("*.parquet"))
    if not files:
        return None

    tables = []
    for fp in files:
        tables.append(pq.read_table(fp))

    merged = pa.concat_tables(tables)
    # 按时间排序、去重
    df = merged.to_pandas()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    out_dir = MERGED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stock}.parquet"
    pq.write_table(pa.Table.from_pandas(df), out_path, compression="zstd")

    logger.info(f"[{stock}] 合并完成: {len(df)} 行 → {out_path}")
    return out_path


# ── 主流程 ─────────────────────────────────────────────
async def main():
    global logger
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="抓取 Torn 股票 M1 历史数据")
    parser.add_argument("--stocks", type=str, default="",
                        help="逗号分隔的股票代码，默认抓全部 35 只")
    parser.add_argument("--years", type=float, default=YEARS,
                        help=f"抓取年数，默认 {YEARS}")
    parser.add_argument("--no-resume", action="store_true",
                        help="不从断点续传，从头开始")
    parser.add_argument("--merge-only", action="store_true",
                        help="只合并已有 chunk，不抓取新数据")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"并发数，默认 {CONCURRENCY}")
    parser.add_argument("--rate", type=int, default=RATE_PER_SEC,
                        help=f"速率限制（次/秒），默认 {RATE_PER_SEC}")
    args = parser.parse_args()

    resume = not args.no_resume
    concurrency = args.concurrency
    rate = args.rate

    logger.info("=" * 60)
    logger.info("Torn Stock M1 数据抓取")
    logger.info(f"  年份范围: {args.years} 年")
    logger.info(f"  并发数: {concurrency}")
    logger.info(f"  速率限制: {rate} req/s")
    logger.info(f"  断点续传: {'开启' if resume else '关闭'}")
    logger.info("=" * 60)

    # ── 创建 HTTP 会话 ──
    connector = aiohttp.TCPConnector(
        limit=concurrency + 5,
        limit_per_host=concurrency + 5,
        ttl_dns_cache=300,
    )
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "tornstock-m1-fetcher/1.0"},
    ) as session:

        # ── 获取股票列表 ──
        if args.stocks:
            stocks = [s.strip().upper() for s in args.stocks.split(",") if s.strip()]
            logger.info(f"使用指定股票: {stocks}")
        else:
            stocks = await fetch_stock_list(session)
            if not stocks:
                logger.error("无法获取股票列表，请用 --stocks 手动指定")
                return

        logger.info(f"目标: {len(stocks)} 只股票")

        # ── 计算时间范围 ──
        end_ts = round_down_to_minute(now_ts())
        start_ts = round_down_to_minute(end_ts - int(args.years * 365.25 * 24 * 3600))
        logger.info(f"时间范围: {ts_to_str(start_ts)} → {ts_to_str(end_ts)}")
        logger.info(f"预计每只股票 {(end_ts - start_ts) // 60} 分钟 ≈ {(end_ts - start_ts) / 60 / CHUNK_SIZE:.0f} 个 chunk")

        # ── 生成任务列表 ──
        all_tasks = []
        for stock in stocks:
            chunks = compute_chunks(start_ts, end_ts)
            for (f, t) in chunks:
                all_tasks.append((stock, f, t))

        total_tasks = len(all_tasks)
        logger.info(f"总任务数: {total_tasks} (≈ {total_tasks / rate / 60:.0f} 分钟预估)")

        # ── 断点续传 ──
        progress = load_progress() if resume else {}
        if progress:
            pending = [(s, f, t) for s, f, t in all_tasks if not chunk_done(progress, s, f, t)]
            logger.info(f"进度恢复: {len(progress)} 已完成, {len(pending)} 待处理")
        else:
            pending = all_tasks
            logger.info(f"全新抓取: {len(pending)} 个任务")

        if args.merge_only:
            logger.info("跳过抓取，仅执行合并")
            pending = []

        # ── 并发抓取 ──
        if pending:
            sem = asyncio.Semaphore(concurrency)
            rate_limiter = RateLimiter(rate)

            success_count = 0
            fail_count = 0
            empty_count = 0

            async def worker(stock: str, from_ts: int, to_ts: int) -> bool:
                nonlocal success_count, fail_count, empty_count
                records = await fetch_chunk(session, sem, rate_limiter, stock, from_ts, to_ts)
                if records is None:
                    fail_count += 1
                    return False
                if not records:
                    empty_count += 1
                else:
                    write_chunk_parquet(stock, from_ts, records)
                    success_count += 1
                mark_chunk_done(progress, stock, from_ts, to_ts)
                return True

            # ── 创建进度条 ──
            pbar = async_tqdm(total=len(pending), desc="抓取进度", unit="chunk")

            # ── 并发执行 ──
            async def run_with_pbar(task):
                result = await worker(*task)
                pbar.update(1)
                pbar.set_postfix({
                    "成功": success_count,
                    "失败": fail_count,
                    "空": empty_count,
                })
                return result

            # 使用 TaskGroup 限制并发
            sem_global = asyncio.Semaphore(concurrency)

            async def bounded_worker(task):
                async with sem_global:
                    return await run_with_pbar(task)

            await asyncio.gather(*[bounded_worker(t) for t in pending])

            pbar.close()

            # ── 最终保存进度 ──
            save_progress(progress)

            logger.info(
                f"抓取完成: 成功 {success_count}, 失败 {fail_count}, "
                f"空数据 {empty_count}, 总计 {len(progress)}"
            )

        # ── 合并每只股票的数据 ──
        logger.info("开始合并各股票数据...")
        for stock in stocks:
            merge_stock(stock)

        # ── 统计 ──
        logger.info("\n" + "=" * 60)
        logger.info("最终统计:")
        total_rows = 0
        for stock in stocks:
            fpath = MERGED_DIR / f"{stock}.parquet"
            if fpath.exists():
                table = pq.read_table(fpath)
                rows = len(table)
                ts_min = ts_to_str(table.column("timestamp")[0].as_py())
                ts_max = ts_to_str(table.column("timestamp")[-1].as_py())
                logger.info(f"  {stock}: {rows:>10,} 行  [{ts_min} → {ts_max}]")
                total_rows += rows
            else:
                logger.warning(f"  {stock}: 无数据")
        logger.info(f"  总计: {total_rows:,} 行")
        logger.info(f"  输出目录: {MERGED_DIR.resolve()}")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
