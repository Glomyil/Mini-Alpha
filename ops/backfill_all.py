"""ops/backfill_all.py —— 补字段监管器：起 N 个子进程 + 汇总进度 + 自动重启

为什么需要它（2026-11-09）
    手开 4 个终端跑分片，有三个问题：
      ① 死一个没人知道（今天实际发生了：报错让进程整个退出）
      ② 进度要一个个窗口数
      ③ 谁跑完了、谁卡住了，看不出来
    这个脚本把三件事包掉：管进程 / 打一行总进度 / 死了重启。

用法
    python -m ops.backfill_all                  # 4 个 worker
    python -m ops.backfill_all --workers 3      # 少开一点（服务端有并发上限，实测 4 稳、>5 会被踢）
    python -m ops.backfill_all --sleep 0.3      # 给每个请求加点间隔，更保守
    python -m ops.backfill_all --dry-run        # 只打印分段计划，不启动

⚠️ 跑之前先确保没有残留的 python.exe 在下载 —— 并发数超了服务端会踢会话，
   症状就是「10001001 用户未登录」+「10002007 网络接收错误」并存。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "storage" / "stock_daily"
LOGS = ROOT / "data" / "logs" / "backfill_all"
MAX_RESTART = 3          # 同一段最多重启几次，防止死循环
POLL_S = 20              # 每多少秒打一行进度


def _codes() -> list:
    sys.path.insert(0, str(ROOT))
    from data.loader import list_codes
    return list(list_codes())


def _done_count() -> tuple:
    """已完成（turn 已是 double）的只数 / 总只数。只读 schema，约 1.5 秒。"""
    fs = sorted(DATA.glob("*.parquet"))
    ok = 0
    for f in fs:
        try:
            s = pq.ParquetFile(f).schema_arrow
            if "turn" in s.names and str(s.field("turn").type) == "double":
                ok += 1
        except Exception:            # noqa: BLE001  坏文件不该让监管器崩
            pass
    return ok, len(fs)


def main() -> None:
    ap = argparse.ArgumentParser(description="补字段监管器")
    ap.add_argument("--workers", type=int, default=4, help="并发数（实测 4 稳，>5 会被踢）")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--sleep", default="0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    codes = _codes()
    n = len(codes)
    w = max(1, args.workers)
    size = (n + w - 1) // w
    shards = [(i * size, min(size, n - i * size)) for i in range(w)]
    shards = [(o, c) for o, c in shards if c > 0]

    done0, total = _done_count()
    logger.info(f"总 {n} 只 · 已完成 {done0} ({done0 / max(total,1) * 100:.1f}%) · {len(shards)} 个分片")
    for i, (off, cnt) in enumerate(shards):
        logger.info(f"  分片 {i}: offset {off:5d} limit {cnt:4d} ({codes[off]} ~ {codes[off + cnt - 1]})")
    if args.dry_run:
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    procs = []          # [(idx, Popen, out_fh, restarts)]
    for i, (off, cnt) in enumerate(shards):
        cmd = [sys.executable, "-m", "data.fetcher.baostock_stock_fetcher",
               "--start", args.start, "--backfill-only",
               "--sleep", args.sleep, "--offset", str(off), "--limit", str(cnt)]
        out = open(LOGS / f"shard{i}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
        procs.append([i, p, out, 0, off, cnt])
        logger.info(f"启动分片 {i} (pid {p.pid})")

    try:
        while True:
            alive = sum(1 for it in procs if it[1].poll() is None)
            done, tot = _done_count()
            el = max(time.time() - t_start, 1.0)
            gained = done - done0
            rate = gained / el * 60
            eta = (tot - done) / rate if rate > 0.01 else 0.0
            logger.info(f"DONE {done}/{tot} ({done / max(tot,1) * 100:.1f}%) | "
                        f"{alive}/{len(procs)} 活跃 | {rate:.1f} 只/分 | ETA {eta / 60:.1f}h")

            # 死掉的拉起来（同一段最多 MAX_RESTART 次）
            for it in procs:
                idx, p, out, rs, off, cnt = it
                if p.poll() is None:
                    continue
                if rs >= MAX_RESTART:
                    logger.error(f"分片 {idx} 已重启 {rs} 次仍失败，放弃（offset {off}）")
                    continue
                logger.warning(f"分片 {idx} 退出(code={p.returncode})，第 {rs + 1} 次重启")
                out.close()
                out = open(LOGS / f"shard{idx}_r{rs + 1}.log", "w", encoding="utf-8")
                cmd = [sys.executable, "-m", "data.fetcher.baostock_stock_fetcher",
                       "--start", args.start, "--backfill-only",
                       "--sleep", args.sleep, "--offset", str(off), "--limit", str(cnt)]
                it[1] = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
                it[2] = out
                it[3] = rs + 1

            if all(it[1].poll() is not None for it in procs) and \
                    all(it[3] >= MAX_RESTART for it in procs if it[1].poll() is not None):
                break
            if alive == 0 and all(it[1].poll() is not None for it in procs):
                # 全部结束且没有可重启的
                if all(it[3] >= MAX_RESTART or it[1].returncode == 0 for it in procs):
                    break
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        logger.warning("收到 Ctrl+C，正在结束子进程…")
    finally:
        for it in procs:
            if it[1].poll() is None:
                it[1].terminate()
                try:
                    it[1].wait(timeout=10)
                except Exception:      # noqa: BLE001
                    it[1].kill()
            try:
                it[2].close()
            except Exception:          # noqa: BLE001
                pass
        done, tot = _done_count()
        logger.info(f"结束 · DONE {done}/{tot} ({done / max(tot,1) * 100:.1f}%) · "
                    f"耗时 {(time.time() - t_start) / 3600:.1f} 小时")
        logger.info(f"子进程日志在 {LOGS}")


if __name__ == "__main__":
    main()
