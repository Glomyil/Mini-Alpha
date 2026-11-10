"""ops/backfill_gaps.py —— 补洞：找出所有「turn 缺列 / 类型不对」的文件，只补这些

和 ops/backfill_all.py 的区别
    backfill_all  分 N 段全量扫一遍（适合首次补齐，几千只）
    backfill_gaps 只扫 schema 找出洞，然后只对洞发请求（适合跑完后的收尾，几十只）
    ⇒ 洞少的时候快得多：5000 只里只有 87 个洞，就不用把 5000 只都过一遍

用法（项目根目录）
    python -m ops.backfill_gaps                 # 补所有洞
    python -m ops.backfill_gaps --dry-run       # 只列出有哪些洞，不发请求
    python -m ops.backfill_gaps --field turn    # 指定补哪个字段

退出码
    0 = 全部补齐
    1 = 仍有洞（会列出代码）

⚠️ 只处理 turn 这一个字段（其余必备列缺失说明数据本身有问题，应当重下该票，
   不该用"补字段"路径悄悄糊过去）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find_gaps() -> list:
    """扫一遍 schema，返回缺 turn（或 turn 不是 double）的代码列表。约 2 秒。"""
    import data.fetcher.baostock_stock_fetcher as F
    gaps = []
    for f in sorted(F.STOCK_DAILY_DIR.glob("*.parquet")):
        try:
            s = pq.ParquetFile(f).schema_arrow
            if "turn" not in s.names or str(s.field("turn").type) != "double":
                gaps.append(f.stem)
        except Exception:                          # noqa: BLE001  坏文件也算洞
            gaps.append(f.stem)
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description="补 turn 字段的洞")
    ap.add_argument("--dry-run", action="store_true", help="只列出洞，不发请求")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    import data.fetcher.baostock_stock_fetcher as F

    t0 = time.time()
    gaps = find_gaps()
    total = len(list(F.STOCK_DAILY_DIR.glob("*.parquet")))
    logger.info(f"文件 {total} 个 · 洞 {len(gaps)} 个（扫描 {(time.time()-t0):.1f} 秒）")
    if not gaps:
        logger.info("没有洞 ✓")
        return 0
    if args.dry_run:
        for c in gaps:
            print("  " + c)
        return 1

    bs = F._login_retry()                          # 带退避重试，不会一次失败就退出
    ok = fail = 0
    failed = []
    try:
        for i, code in enumerate(gaps, 1):
            p = F.STOCK_DAILY_DIR / f"{code}.parquet"
            if not p.exists():
                fail += 1
                failed.append(code)
                continue
            miss = F.missing_fields(p)
            if not miss:
                ok += 1
                continue
            for attempt in (1, 2):
                try:
                    r = F.backfill_one(bs, code, p, miss)
                    # backfilled_nan：字段写进去了但整列是 NaN（如全程停牌）
                    #   ⇒ 也算补完，否则会永远被当"洞"重试
                    if r.get("status") in ("backfilled", "backfilled_nan"):
                        ok += 1
                    else:
                        fail += 1
                        failed.append(code)
                        logger.warning(f"{code}: {r.get('status')}")
                    break
                except Exception as e:             # noqa: BLE001
                    if F._is_session_error(e) and attempt == 1:
                        logger.warning(f"{code} 会话失效 ⇒ 重连重试")
                        with contextlib.redirect_stdout(io.StringIO()):
                            try:
                                F._logout(bs)
                            except Exception:      # noqa: BLE001
                                pass
                        bs = F._login_retry()
                        continue
                    fail += 1
                    failed.append(code)
                    logger.error(f"{code}: {type(e).__name__}: {str(e)[:80]}")
                    break
            if i % 10 == 0 or i == len(gaps):
                logger.info(f"{i}/{len(gaps)} | ok={ok} fail={fail} | {(time.time()-t0)/60:.1f} 分钟")
            time.sleep(0.2)
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                F._logout(bs)
            except Exception:                      # noqa: BLE001
                pass

    left = find_gaps()
    logger.info(f"完成: ok={ok} fail={fail} · 剩余洞 {len(left)} 个 · 耗时 {(time.time()-t0)/60:.1f} 分钟")
    if left:
        logger.error("仍有洞: " + ", ".join(left[:20]) + (" ..." if len(left) > 20 else ""))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
