from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import constants as C
from data import loader
from data.cleaner import pipeline


def store_one(df, dest_dir: Path, code: str) -> int:
    """原子写一只：先写 .parquet.tmp，成功后 os.replace 改名；返回字节数。

    同盘 rename 是原子的 ⇒ 目标文件要么是旧的、要么是新的，永远不会是半个。
    被打断时最多留一个 .tmp（可删），不会污染正式文件。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f"{code}.parquet.tmp"
    final = dest_dir / f"{code}.parquet"
    df.to_parquet(tmp, index=False, compression=C.PARQUET_COMPRESSION)
    os.replace(tmp, final)
    return final.stat().st_size


def source_fingerprint(files: list[Path]) -> dict:
    """记录"这批净数据是从哪批源数据来的" —— 三个便宜且够用的量（只读元数据，不读内容）。"""
    if not files:                                        # 空清单：max() 会 ValueError，提前返回
        return {"files": 0, "max_mtime": None, "total_mb": 0.0}
    return {
        "files": len(files),
        "max_mtime": datetime.fromtimestamp(
            max(p.stat().st_mtime for p in files)).strftime("%Y-%m-%d %H:%M:%S"),
        "total_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1),
    }


def run(codes: list[str], start: str, end: str, out_dir: Path, *,
        src_dir: Path | None = None, steps=None, every: int = 1000) -> dict:

    src_dir = Path(src_dir) if src_dir else C.STOCK_DAILY_DIR
    out_dir = Path(out_dir)

    # —— 段级检查：这三条不满足就没法开工，直接 raise（不要"跳过继续" ✗）——
    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在：{src_dir}")
    if not codes:
        raise RuntimeError("候选清单为空 —— 检查源目录 / 区间，别拿空清单往下跑")
    out_dir.mkdir(parents=True, exist_ok=True)          # 建不出来会自己抛 ✓

    # 源指纹要在开跑前取（跑完后源目录可能已经变了）
    src_files = [src_dir / f"{c}.parquet" for c in codes]
    agg = {"stocks": 0, "rows": 0, "bytes": 0}
    failures: list[str] = []
    written: list[str] = []
    empty = 0
    cols: dict[str, set] = {}

    ctx_base = {"universe": "stock", "asset_type": "stock"}
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        try:
            # ① 读盘也在 try 里 ⇒ 坏文件只影响这一只 ✓
            df = loader.load_one(code, start, end, src_dir)
            if df is None:                                    # ② 空数据 = 业务现象（未上市/整段停牌）✓
                empty += 1
                continue

            out, reports = pipeline.run_one(                  # ③ 执行器只叫号：契约 + 逐环节 + 报告卡 ✓
                df, {**ctx_base, "code": code}, steps)

            n = store_one(out, out_dir, code)                 # ④ 原子写 + 计数
            agg["stocks"] += 1
            agg["rows"] += len(out)
            agg["bytes"] += n
            written.append(code)
            for r in reports:                                 # ⑤ 各环节新增列汇总（= 登记表的自动核对 ✓）
                if r["columns_added"]:
                    cols.setdefault(r["name"], set()).update(
                        r["columns_added"])
        except Exception as e:                                # noqa: BLE001 —— 只级：不中断全库 ✓
            failures.append(f"{code}: {type(e).__name__}: {e}")
            logger.warning(f"{code} 处理失败，已跳过：{type(e).__name__}: {e}")
            continue

        if every and i % every == 0:                          # ⑥ 进度（不做耗时统计 ✓）
            print(f"[{i:>6}/{len(codes)}] 已写 {agg['stocks']:>5} 只 "
                  f"{agg['rows']:>10,} 行")

    # —— 完整性不变量：三个计数互斥且穷尽 ⇒ 对不上说明有票被静默漏掉了 ——
    assert len(written) + len(failures) + empty == len(codes), (
        f"计数对不上：written {len(written)} + failures {len(failures)} "
        f"+ empty {empty} ≠ codes {len(codes)}")

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interval": [start, end],
        "n_codes": len(codes),
        "stocks": agg["stocks"], "rows": agg["rows"],
        "bytes": agg["bytes"], "mb": round(agg["bytes"] / 2**20, 1),
        "empty": empty,                                  # 不算失败 ✓
        "failures": failures,
        "written": written,
        "steps": [n for n, _ in (pipeline.PIPELINE if steps is None else steps)],
        "source_fingerprint": source_fingerprint(src_files),
        "elapsed_s": round(time.time() - t0, 1),         # 只进 meta，不打印 ✓
    }
    return meta


def store_meta(meta: dict, out_dir: Path, meta_p: Path | None = None) -> Path:
    """总账【最后】写：它的存在 = "本次全套跑完" ✓（默认 <out_dir 上级>/cleaner_meta.json）

    唯一的写盘入口 ⇒ 命令行入口(main) 与总调度(run_all) 都调它，
    账本格式与"最后写"这条不变量都只有一处定义 ✓
    """
    meta_p = Path(meta_p) if meta_p else Path(out_dir).parent / "cleaner_meta.json"
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return meta_p


def print_summary(meta: dict, out_dir: Path, meta_p: Path) -> None:
    print("-" * 100)
    print(
        f"合计：{meta['stocks']:,} 只 | {meta['rows']:,} 行 | {meta['mb']} MB  →  {out_dir}")
    if meta["failures"]:
        print(f" 失败 {len(meta['failures'])} 只：{meta['failures'][:5]}")
    else:
        print(" 落盘完成，无失败")
    print(f"区间内无数据 {meta['empty']} 只（未上市 / 整段停牌，不算失败）")
    print(f"总账已写：{meta_p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="清洗 → 净数据落盘")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 只")
    ap.add_argument("--codes", default=None, help="逗号分隔的 code 清单")
    ap.add_argument("--src", default=None, help="源目录（默认 STOCK_DAILY_DIR）")
    ap.add_argument("--out", default=None, help="目标目录（默认 CLEANER_DAILY_DIR）")
    ap.add_argument("--meta", default=None,
                    help="总账路径（默认 <目标目录上级>/cleaner_meta.json）")
    ap.add_argument("--every", type=int, default=1000, help="每 N 只打一行进度")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args()

    out_dir = Path(a.out) if a.out else C.CLEANER_DAILY_DIR
    meta_p = Path(a.meta) if a.meta else out_dir.parent / "cleaner_meta.json"
    src_dir = Path(a.src) if a.src else C.STOCK_DAILY_DIR

    all_codes = loader.list_codes(src_dir)
    if a.codes:
        want = {c.strip() for c in a.codes.split(",") if c.strip()}
        codes = [c for c in all_codes if c in want]
    else:
        codes = all_codes[: a.limit] if a.limit else all_codes

    print(f"源 {src_dir}   输出 {out_dir}   总账 {meta_p}")
    print(f"标的 {len(codes)}/{len(all_codes)} 只 | 区间 {a.start} ~ {a.end} | 环节 "
          + " → ".join(n for n, _ in pipeline.PIPELINE))
    print("-" * 100)

    meta = run(codes, a.start, a.end, out_dir,
               src_dir=src_dir, every=a.every)

    # 总账最后写：它的存在 = "本次全套跑完" ✓（唯一入口 = store_meta ✓ 总调度也用同一个 ✓）
    meta_p = store_meta(meta, out_dir, meta_p)

    print_summary(meta, out_dir, meta_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
