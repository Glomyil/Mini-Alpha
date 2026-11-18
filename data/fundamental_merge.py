"""fundamental_merge —— 五张季频财务表 -> 每股事件宽表（纯搬运 + 类型转换，不做判断）。

输入  data/storage/fundamental/{table}/{code}.parquet
输出  data/storage/fundamental_event/{code}.parquet
      每股 47 行 x 39 列 = statDate + 5 个 pubDate_<表名> + pubDate_max + 32 指标

口径（定死，改动前先读）:
  1. 唯一键 (code, statDate)；五路 FULL OUTER JOIN —— 行数实测不齐（27/36/47），
     INNER 会把「某表那期还没披露」整行丢掉。
  2. 五个 pubDate **各自取别名、全部保留**。它们在 99.99% 的期次上相同，看起来是冗余；
     但剩下 0.01% 里差异本身就是唯一的信息 —— 回答「这期为什么晚了、是哪张表拖的」。
     （实测 sh.600018 2019-12-31：快报 2020-01-10 / 年报 2020-04-10，差 91 天。PITFALLS #31）
  3. pubDate_max = 五者的 greatest()，是**可见性唯一依据**。
     用 least() 会让上面那期提前 91 天可见 = 前视，而且不报错。
  4. 产物无 index、无 code 列（身份靠文件名，与 cleaner_daily 同构）。
  5. **日期 CAST 成 TIMESTAMP、指标 TRY_CAST 成 DOUBLE**（2026-11-18 修）。
     baostock 落盘时所有指标都是字符串，缺失值是【空串】不是 NULL —— 不转的话
     isna() 恒为 0（覆盖率统计会骗人），且下游 df[x > 0] 直接 TypeError。
     TRY_CAST 把空串变成 NULL，正好。
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
from loguru import logger

from config.constants import FUNDAMENTAL_DIR

EVENT_DIR = FUNDAMENTAL_DIR.parent / "fundamental_event"

# 表名 -> 该表指标列（共 32 个）。逐列生成 TRY_CAST，所以用 list 不用逗号串。
TABLES: dict[str, list[str]] = {
    "profit": ["roeAvg", "npMargin", "gpMargin", "netProfit",
               "epsTTM", "MBRevenue", "totalShare", "liqaShare"],
    "growth": ["YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"],
    "balance": ["currentRatio", "quickRatio", "cashRatio",
                "YOYLiability", "liabilityToAsset", "assetToEquity"],
    "operation": ["NRTurnRatio", "NRTurnDays", "INVTurnRatio",
                  "INVTurnDays", "CATurnRatio", "AssetTurnRatio"],
    "cashflow": ["CAToAsset", "NCAToAsset", "tangibleAssetToAsset",
                 "ebitToInterest", "CFOToOR", "CFOToNP", "CFOToGr"],
}

# 小于这个字节数的 parquet 视作「半写文件」跳过，而不是让整批失败。
# 根因：baostock_fundamental_fetcher.py 的 df.to_parquet(target) 是【非原子】写，
# 下载器在跑时读的人会撞上 0 字节文件（实测 balance/sh.601006.parquet）。
MIN_BYTES = 200


def _files(table: str, src: Path) -> list[str]:
    """该表下可用的 parquet 路径清单（POSIX 形式，供 read_parquet 用）。"""
    return [p.as_posix() for p in sorted((src / table).glob("*.parquet"))
            if p.stat().st_size > MIN_BYTES]


def _build_sql(src: Path) -> str:
    """五表 -> 一条 SQL：五个 CTE 各读一张表，再用 FULL OUTER JOIN 拼起来。"""
    ctes, aliases = [], []
    for i, (t, cols) in enumerate(TABLES.items()):
        a = f"t{i}"                       # t0..t4，避免 p/g/b/o/c 撞 SQL 关键字
        files = _files(t, src)
        if not files:
            raise FileNotFoundError(f"{src / t} 下没有可用的 parquet")
        # 指标逐列 TRY_CAST：parquet 里是字符串、缺失是空串，
        # 普通 CAST 遇空串会抛，TRY_CAST 给 NULL —— 正是我们要的 NaN。
        proj = ", ".join(f"TRY_CAST({c} AS DOUBLE) AS {c}" for c in cols)
        ctes.append(
            f"{a} AS (SELECT code, CAST(statDate AS TIMESTAMP) AS statDate, "
            f"CAST(pubDate AS TIMESTAMP) AS pubDate_{t}, {proj} "
            f"FROM read_parquet({files!r}))")
        aliases.append(a)
    head = ",\n     ".join(ctes)
    # 每个 JOIN 都必须带 USING (code, statDate) —— 漏了会变成无条件 JOIN，
    # DuckDB 要么报错要么退化成笛卡尔积。
    body = "\n  FULL OUTER JOIN ".join(
        f"{a} USING (code, statDate)" for a in aliases[1:])
    return f"WITH {head}\nSELECT * FROM {aliases[0]}\n  FULL OUTER JOIN {body}"


def build_events(out: Path = EVENT_DIR, *, src: Path | None = None) -> dict:
    """跑完整条：合并 -> 加 pubDate_max -> 按 code 拆文件。返回统计字典。"""
    src = Path(src or FUNDAMENTAL_DIR)
    out = Path(out)
    t_start = time.time()

    con = duckdb.connect()
    con.execute("CREATE TABLE ev AS " + _build_sql(src))
    con.execute("""CREATE TABLE ev2 AS SELECT *,
      greatest(pubDate_profit, pubDate_growth, pubDate_balance,
               pubDate_operation, pubDate_cashflow) AS pubDate_max
    FROM ev""")
    n_rows = con.execute("SELECT count(*) FROM ev2").fetchone()[0]

    out.mkdir(parents=True, exist_ok=True)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT code FROM ev2 ORDER BY code").fetchall()]
    failures: list[str] = []
    for c in codes:
        try:
            con.execute(
                "COPY (SELECT * EXCLUDE (code) FROM ev2 WHERE code='" + c +
                "' ORDER BY statDate) TO '" + str(out / (c + '.parquet')) +
                "' (FORMAT PARQUET)")
        except Exception as e:          # 只级失败：不中断全批，记下来继续
            failures.append(f"{c}: {type(e).__name__}: {e}")
            logger.warning(f"{c} 写出失败，已跳过：{e}")
    con.close()

    meta = {"stocks": len(codes), "rows": int(n_rows), "failures": failures,
            "elapsed_s": round(time.time() - t_start, 1), "out": str(out)}
    logger.info(f"fundamental_event: {meta}")
    if failures:
        logger.warning(f"有 {len(failures)} 只票写出失败（重跑即可，幂等）")
    return meta


if __name__ == "__main__":
    print(build_events())