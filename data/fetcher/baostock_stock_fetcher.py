# -*- coding: utf-8 -*-
"""
A股全量(含退市)日线下载器 —— baostock 官方复权因子版
====================================================
对应方案文档: 股票数据下载方案.md (v0, 已获用户确认参数)

设计要点:
  1. 池子: baostock query_stock_basic() 中 type=='1' 的全部股票(含 status='0' 退市股)
  2. 每只股票落盘单文件 Parquet: raw OHLCV + tradestatus + isST + back_adj_factor
     (后复权价 = raw × back_adj_factor, 因子按事件日 ffill)
  3. baostock 非线程安全 -> 多进程; 2026-09-07 修复:
     弃用 multiprocessing.spawn(Windows cmd 下 WinError 87),
     改用 subprocess 启动 N 个独立子进程(每进程独立 bs.login(), 处理一段代码清单)
  4. 断点续传: 文件存在且末日 >= 目标末日 -> 跳过
  5. 失败重试 3 次 + 指数退避; 失败记入 JSONL 日志, 不中断整体
  6. 因子表查询从 1990-01-01 开始: 保证拿到"起始日之前最近一次事件"的累计因子,
     否则 2015 起点前的历史分红会丢(金融逻辑: 因子是累计值, 不是当期值)
  7. 【合规 2026-09-07】baostock 官方规则: 每日请求 <=5万次、禁止并发连接访问。
     本下载器强制单进程串行(workers 恒=1); 全量约 1.1 万次请求, 远低于日上限;
     黑名单/冷却期间停手, 不要反复重试(会延长封禁)。

用法(示例):
  python data/fetcher/baostock_stock_fetcher.py                      # 全量(5553), 单进程串行
  python data/fetcher/baostock_stock_fetcher.py --codes sh.600000,sh.600068   # 调试(单只)
  python data/fetcher/baostock_stock_fetcher.py --limit 500 --offset 3000     # 分片续跑(每晚一片)

输出:
  data/storage/stock_daily/{code}.parquet
  data/storage/stock_meta.json
  data/logs/stock_download_YYYYmmdd_HHMMSS*.jsonl   (父+每个子进程各一份)
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# 路径(引用 config.constants 已有常量; 不修改 constants.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.constants import (  # noqa: E402
    LOG_DIR, MAX_RETRIES, PARQUET_COMPRESSION, REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE, STOCK_DAILY_DIR, STOCK_META_FILE,
)

# 全局 socket 超时: 网络闪断时 baostock 可能"静默挂死"(不报错也不返回),
# 设超时让死连接快速失败 → 走重试/记 error, 而不是无限等待(2026-09-08 事故修复)。
socket.setdefaulttimeout(REQUEST_TIMEOUT)

# 路径一律来自 config.constants（2026-09-10：删掉本地重复定义，避免两处维护漂移）

# 因子查询起点: 必须早于行情起点, 才能拿到起始日前最近事件的累计因子
FACTOR_QUERY_START: str = "1990-01-01"

# ---------------------------------------------------------------------------
# 默认参数(类属性风格, 与 QD 规则一致: 可被 CLI/yml 覆盖)
# 【baostock 官方规则, 用户确认 2026-09-07】
#   1) 每日 API 请求不能超过 5 万次, 超过进入黑名单控制;
#   2) 禁止并发连接访问。
# => 本下载器强制: 单进程串行(workers 恒=1), 请求间隔 0.6s ± 抖动;
#    全量 5553 只 ≈ 每只 2 次查询(行情+因子) ≈ 1.1 万次/日 << 5 万上限。
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "start_date": "2015-01-01",     # 用户已确认: 与 ETF 数据区间对齐
    "workers": 1,                   # 官方禁止并发 → 恒单进程串行
    "sleep": 0.6,                   # 单只股票间的请求间隔(秒), 含 ±50% 抖动
}

KLINE_FIELDS: str = (
    # 2026-11-04 加 turn（换手率）：流通股本 = volume / (turn/100)，
    # 流通市值 = close(未复权) × 流通股本。实测 sh.600000 反推 293.44 亿股
    # vs 实际 293.52 亿股（误差 0.03%），且年内 CV ≈ 0.0001。
    # ⚠️ 加字段必须【全量重下】—— --update 增量只会把 turn 填在最后几天，
    #    历史行 concat 后全是 NaN（已实测推演，见 T-6.1 工单）。
    "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST"
)

# 交易状态/ST 为空字符串时的兜底值(非价格类字段, 允许补 0)
_MISSING_STATUS_FILL: int = 0


# ============================================================================
# baostock 会话(每进程独立 login/logout)
# ============================================================================

def _login() -> Any:
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
    return bs


def _logout(bs: Any) -> None:
    try:
        bs.logout()
    except Exception:
        pass


def fetch_kline(bs: Any, code: str, start: str, end: str,
                fields: str = KLINE_FIELDS) -> pd.DataFrame:
    """不复权日线(adjustflag=3), 含 tradestatus/isST

    fields: 默认全部 KLINE_FIELDS；补字段时只查需要的那几个（见 backfill_one）
    """
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end,
        frequency="d", adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"kline error {rs.error_code}: {rs.error_msg}")
    rows: List[List[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    expected_cols = fields.split(",")
    df = pd.DataFrame(rows)
    if df.shape[1] != len(expected_cols):
        # 防御: 畸形返回(网络损坏/服务端异常)时给出可诊断信息, 而不是 pandas 的抽象报错
        raise RuntimeError(
            f"kline 返回列数异常: 期望 {len(expected_cols)} 列, 实际 {df.shape[1]} 列; "
            f"code={code}, 首行={rows[0]!r}"
        )
    df.columns = expected_cols
    # 类型: 价格/量/换手 -> float(空串转 NaN); 状态 -> int(空串补 0)
    #   用 in df.columns 判断，这样补字段时只带 close,volume,turn 也不会 KeyError
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tradestatus", "isST"]:
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce")
                       .fillna(_MISSING_STATUS_FILL).astype("int8"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_factor(bs: Any, code: str, end: str) -> pd.DataFrame:
    """官方复权因子表: 只在除息/折算日有行; 列=backAdjustFactor(累计后复权因子)"""
    rs = bs.query_adjust_factor(code=code, start_date=FACTOR_QUERY_START, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"adjust_factor error {rs.error_code}: {rs.error_msg}")
    fields: List[str] = list(rs.fields)
    rows: List[List[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=["date", "back_adj_factor"])
    if len(fields) != len(rows[0]):
        raise RuntimeError(
            f"adjust_factor 返回列数异常: fields={fields}, 首行={rows[0]!r}, code={code}"
        )
    df = pd.DataFrame(rows, columns=fields)
    df["date"] = pd.to_datetime(df["dividOperateDate"], errors="coerce")
    df["back_adj_factor"] = pd.to_numeric(df["backAdjustFactor"], errors="coerce")
    df = df.dropna(subset=["date", "back_adj_factor"]).sort_values("date").reset_index(drop=True)
    return df[["date", "back_adj_factor"]]


# ============================================================================
# 单只股票处理
# ============================================================================

def process_one(bs: Any, code: str, start: str, end: str, target: Path) -> Dict[str, Any]:
    """拉取并落盘单只股票; 返回结果统计 dict"""
    kdf, fdf, attempt = None, None, None
    for attempt_i in range(MAX_RETRIES):
        try:
            kdf = fetch_kline(bs, code, start, end)
            fdf = fetch_factor(bs, code, end)
            break
        except Exception as e:  # noqa: BLE001  -- 重试循环需捕获全部网络/接口异常
            attempt = e
            if attempt_i < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt_i + 1)))  # 2s, 4s 退避
    if kdf is None or fdf is None:
        return {"code": code, "status": "error",
                "error": f"{type(attempt).__name__}: {attempt}",
                "traceback": traceback.format_exc()}

    if kdf.empty:
        return {"code": code, "status": "no_data_in_window", "rows": 0}

    # 因子合并: 事件日前无因子 -> 1.0(从未分红的新股), 或前向填充到起始日前最近事件值
    if fdf.empty:
        kdf["back_adj_factor"] = 1.0
    else:
        merged = pd.merge_asof(
            kdf.sort_values("date"), fdf.sort_values("date"),
            on="date", direction="backward",
        )
        kdf["back_adj_factor"] = merged["back_adj_factor"].fillna(1.0)

    kdf["code"] = code
    kdf["back_adj_factor"] = kdf["back_adj_factor"].astype("float64")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    kdf.to_parquet(tmp, compression=PARQUET_COMPRESSION)
    tmp.replace(target)
    return {
        "code": code, "status": "ok", "rows": len(kdf),
        "date_min": str(kdf["date"].min().date()),
        "date_max": str(kdf["date"].max().date()),
        "factor_events": len(fdf),
    }


# ============================================================================
# 增量更新(2026-09-09 新增): 只追加新交易日, 不重下历史
# ============================================================================

# ============================================================================
# 补字段(2026-11-09 新增): 给【已有文件】补上后来才加进 KLINE_FIELDS 的列
#
#   为什么需要：turn 是在 5476 个文件已经落盘之后才加的。
#     整段重下 = 15 小时（还容易被限流）；
#     只查缺的那一个字段 = 约 5 小时，而且【其余字段一个字节都不动】。
#
#   设计原则（很重要）：
#     ① 只查缺的字段，不重下已有字段 —— 已有数据是已验收的资产，不冒被替换的风险
#     ② 多带一个 close 作【对齐校验】：新查的 close 必须和原文件逐行一致，
#        不一致说明日期错位或数据源改了历史 ⇒ 当场报错，绝不静默合并
#     ③ 日期范围用【原文件自己的】范围，不拉长也不缩短
# ============================================================================

_REQUIRED_FIELDS: tuple = tuple(KLINE_FIELDS.split(",")) + ("back_adj_factor",)


_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume", "amount", "turn")


def missing_fields(target: Path) -> List[str]:
    """这个文件缺哪些必备列（只读 parquet 的 schema，不读数据，很快）。

    ⚠️ 「存在但类型不对」也算缺 —— 实测踩过：
       2026-11-04 第一次只把 turn 加进 KLINE_FIELDS，忘了加进 fetch_kline 的数值化列表
       ⇒ 那批文件的 turn 是 object(字符串)
       ⇒ 后面 update_one 做 pd.concat([object, float]) 时 pyarrow 直接报
         ArrowTypeError: Expected bytes, got a 'float' object
       ⇒ 所以这里把 object 型的数值列也判为"要修"，让它走同一条补字段路径被自动修好。
    """
    if not target.exists():
        return list(_REQUIRED_FIELDS)
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(target).schema_arrow
        have = set(schema.names)
    except Exception:                     # noqa: BLE001  文件损坏 → 当作全缺
        return list(_REQUIRED_FIELDS)
    bad = [c for c in _NUMERIC_FIELDS
           if c in have and str(schema.field(c).type) in ("string", "large_string")]
    return [c for c in _REQUIRED_FIELDS if c not in have] + bad


def backfill_one(bs: Any, code: str, target: Path, missing: List[str]) -> Dict[str, Any]:
    """给已有文件补缺失字段，只查这些字段 + close(校验)，其余列原样保留。"""
    existing = pd.read_parquet(target)
    # 存在但类型不对的列（如 object 型 turn）先丢掉，合并时会以正确类型重建
    _drop = [c for c in missing if c in existing.columns]
    if _drop:
        existing = existing.drop(columns=_drop)
    d = pd.to_datetime(existing["date"])
    d0, d1 = str(d.min().date()), str(d.max().date())

    # 要查的字段：缺的 + close(仅作校验，若 close 本身缺席就靠它补)
    want = sorted(set(missing) | {"close"})
    df = fetch_kline(bs, code, d0, d1, fields="date,code," + ",".join(want))
    if df.empty:
        return {"code": code, "status": "backfill_no_data", "missing": missing}

    # ── 对齐校验：close 必须逐行一致 ──
    if "close" not in missing:
        old = existing.assign(_d=d)[["_d", "close"]]
        new = df[["date", "close"]].rename(columns={"date": "_d"})
        cmp = old.merge(new, on="_d", how="inner", suffixes=("_old", "_new")).dropna()
        if len(cmp) == 0:
            return {"code": code, "status": "backfill_no_overlap", "missing": missing}
        tol = cmp["close_old"].abs().clip(lower=1e-9) * 1e-9
        bad = int(((cmp["close_old"] - cmp["close_new"]).abs() > tol).sum())
        if bad:
            raise RuntimeError(
                f"{code} 补字段对齐失败：{bad}/{len(cmp)} 行 close 与原文件不一致 "
                f"⇒ 日期错位或数据源改了历史，拒绝合并（绝不静默写坏数据）")

    # ── 合并：只把缺失列按 date 贴上去 ──
    #   ⚠️ add 里【不能带 date 列】—— existing 也有 date，合并时会给两边都加后缀
    #      （date_x / date_y），把原文件的 date 列弄坏。只带 _d 作连接键。
    add = df[[c for c in missing if c in df.columns]].copy()
    add["_d"] = df["date"].to_numpy()
    if existing["date"].duplicated().any():
        raise RuntimeError(f"{code} 原文件 date 有重复 ⇒ 拒绝合并（merge 会炸行）")
    out = existing.assign(_d=d.to_numpy()).merge(add, on="_d", how="left")
    out = out.drop(columns=["_d"]).sort_values("date").reset_index(drop=True)

    # ⚠️「整列全是 NaN」也要写进去 —— 否则这条永远补不完：
    #   实测 sh.601268 全程停牌（volume=0），baostock 的 turn 全是空串
    #   ⇒ 没有换手率就算不出流通股本 ⇒ NaN 就是【正确答案】，不是失败
    #   如果不写，schema 永远缺 turn ⇒ 以后每次运行都把它当"洞"重试，永远补不完 ✗
    got = [c for c in missing if c in out.columns]
    still = [c for c in missing if c not in got]
    if still:
        return {"code": code, "status": "backfill_missing", "missing": still}

    all_nan = [c for c in got if not out[c].notna().any()]
    tmp = target.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, compression=PARQUET_COMPRESSION)
    tmp.replace(target)
    return {"code": code,
            "status": "backfilled_nan" if all_nan else "backfilled",
            "fields": got, "rows": len(out), "all_nan_fields": all_nan}


def _merge_new_rows(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """把新行并入旧表: 拼接 -> 同日去重(保留新值) -> 按日期排序。纯函数, 可离线测试。"""
    both = pd.concat([existing, new], ignore_index=True)
    both = (both.drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True))
    return both


def update_one(bs: Any, code: str, start: str, end: str, target: Path) -> Dict[str, Any]:
    """增量更新单只股票: 只拉 [已有末日, end] 的新 K 线, 追加进原文件。

    - 文件不存在 -> 退回全量 process_one(顺手补缺口)
    - 已有末日 >= end -> skipped(无需更新)
    - 区间内拉不到新行 -> up_to_date(不写盘)
    安全性: 后复权因子是"从上市起累计"的历史常量, 新增除息只会追加一行事件,
    因此只追加新行、不动历史行是正确的。
    """
    if not target.exists():
        r = process_one(bs, code, start, end, target)
        r["mode"] = "full(fill_missing)"
        return r

    existing = pd.read_parquet(target)
    if existing.empty:
        r = process_one(bs, code, start, end, target)
        r["mode"] = "full(empty_file)"
        return r

    last_date = pd.Timestamp(existing["date"].max()).strftime("%Y-%m-%d")
    if last_date >= end:
        return {"code": code, "status": "skipped", "mode": "update",
                "rows": len(existing), "date_max": last_date}

    kdf, fdf, attempt = None, None, None
    for attempt_i in range(MAX_RETRIES):
        try:
            # 从"已有末日(含)"开始拉: 覆盖最后一天, 以防当日数据被事后修正
            kdf = fetch_kline(bs, code, last_date, end)
            fdf = fetch_factor(bs, code, end)
            break
        except Exception as e:  # noqa: BLE001
            attempt = e
            if attempt_i < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt_i + 1)))
    if kdf is None or fdf is None:
        return {"code": code, "status": "error", "mode": "update",
                "error": f"{type(attempt).__name__}: {attempt}",
                "traceback": traceback.format_exc()}

    if kdf.empty:
        return {"code": code, "status": "up_to_date", "mode": "update",
                "rows": len(existing), "date_max": last_date}

    # 因子合并(只对新行做)
    if fdf.empty:
        kdf["back_adj_factor"] = 1.0
    else:
        merged_factor = pd.merge_asof(
            kdf.sort_values("date"), fdf.sort_values("date"),
            on="date", direction="backward",
        )
        kdf["back_adj_factor"] = merged_factor["back_adj_factor"].fillna(1.0)

    kdf["code"] = code
    kdf["back_adj_factor"] = kdf["back_adj_factor"].astype("float64")
    for col in existing.columns:                 # 列对齐(旧文件列顺序可能不同)
        if col not in kdf.columns:
            kdf[col] = pd.NA
    kdf = kdf[existing.columns]
    combined = _merge_new_rows(existing, kdf)

    tmp = target.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, compression=PARQUET_COMPRESSION)
    tmp.replace(target)
    return {"code": code, "status": "updated", "mode": "update",
            "rows": len(combined), "new_rows": len(combined) - len(existing),
            "date_max": str(combined["date"].max().date())}


# ============================================================================
# 代码清单处理(单进程; 由调试模式直接调用, 或由父进程以子进程拉起)
# ============================================================================

def _login_retry(retries: int = 8, base_sleep: float = 1.0) -> Any:
    """带指数退避的重试登录。
    baostock 对高并发会踢会话/临时拒连(10001001/10002007/10057)，
    因此重试次数要多、退避要长(封顶 15s/次), 才能扛过服务器繁忙窗口。
    """
    last: Exception | None = None
    for i in range(retries):
        try:
            return _login()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(base_sleep * (2 ** i), 15.0))
    raise RuntimeError(f"baostock login failed after {retries} retries: {last}")


def _is_session_error(msg) -> bool:
    """baostock 会话失效标志(10001001=用户未登录)

    ⚠️ 必须容忍【异常对象】：调用方有两类
         · r.get("error", "")   → str
         · except ... as e      → RuntimeError 等异常实例
       2026-11-09 就因为这个：

         return "10001001" in msg          # msg 是 RuntimeError
         TypeError: argument of type 'RuntimeError' is not iterable

       ⇒ 异常没人接 ⇒ 整个分片进程退出 ⇒ 表现成"十分钟重启两次"
    """
    s = str(msg)
    return "10001001" in s or "未登录" in s


def run_codes(codes: List[str], start: str, end: str, resume: bool,
              log_path: Path, label: str, sleep_s: float = 0.6,
              update: bool = False, backfill_only: bool = False) -> Dict[str, int]:
    """在当前进程内处理 codes; 结果逐行写入 log_path(JSONL); 返回状态计数
    原则: 登录/网络失败绝不崩溃子进程, 一律落 error 行, 交给"重跑续传"兜底。
    sleep_s: 每只股票之间的间隔(秒), 实际应用 ±50% 随机抖动, 降低风控特征。
    update:  True=增量模式(只追加新交易日, 见 update_one); False=全量回补模式。
    """
    try:
        bs = _login_retry()
    except Exception as e:  # noqa: BLE001
        # 服务器长时间拒连: 整段标记 error 并优雅退出(父进程汇总, 重跑续传)
        with open(log_path, "w", encoding="utf-8") as logf:
            for code in codes:
                logf.write(json.dumps(
                    {"code": code, "status": "error", "error": f"login failed: {e}"},
                    ensure_ascii=False) + "\n")
        print(f"[{label}] 登录失败, 整段 {len(codes)} 只记为 error", flush=True)
        return {"error": len(codes)}
    # `end` 默认 = 日历今天，而数据最多到最后一个交易日。
    #   周末/节假日/盘中跑时，今天 > 最后交易日 ⇒ 用今天比会让所有文件都"不完整"。
    #   ⇒ 改成和【全库已有文件的最大日期】比（= 已知的市场最后交易日）。
    eff_end = end
    if resume and not backfill_only:      # backfill-only 不碰日期维度，不必扫全库算市场末日
        _mx = None
        for _p in STOCK_DAILY_DIR.glob("*.parquet") if STOCK_DAILY_DIR.exists() else []:
            try:
                _d = pd.read_parquet(_p, columns=["date"])["date"].max()
                _mx = _d if _mx is None else max(_mx, _d)
            except Exception:              # noqa: BLE001  坏文件不该拖垮整轮
                continue
        if _mx is not None:
            eff_end = str(pd.Timestamp(_mx).date())

    summary: Dict[str, int] = {}
    consec_errors: int = 0   # 连续失败计数 -> 触发自动冷却(网络风暴期别硬冲)
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            _t_start = time.time()
            for i, code in enumerate(codes):
                # 进度打印放在循环【开头】，不能用 continue 跳过（之前放在末尾，
                # backfill 分支一 continue 就看不到任何进度了 —— 13 小时的盲跑）
                if (i + 1) % 20 == 1 or (i + 1) % 20 == 0:
                    _el = max(time.time() - _t_start, 1.0)   # 防"第一只还没跑完就除零"
                    _rate = (i + 1) / _el
                    _eta = (len(codes) - i - 1) / _rate / 60
                    print(f"[{label}] {i + 1}/{len(codes)} | {summary} | "
                          f"{_rate * 60:.1f} 只/分 | 剩余 {_eta:.0f} 分", flush=True)
                target = STOCK_DAILY_DIR / f"{code}.parquet"
                if update:
                    # 增量模式: 只追加新交易日(文件缺失时自动退回全量)
                    r = update_one(bs, code, start, end, target)
                    if r.get("status") == "error" and _is_session_error(r.get("error", "")):
                        _logout(bs)
                        bs = _login_retry()
                        r = update_one(bs, code, start, end, target)
                else:
                    if resume and target.exists():
                        miss = missing_fields(target)
                        if miss:
                            # 缺字段 ⇒ 只补缺的字段（不重下历史）
                            # ⚠️ 必须包 try：fetch_kline 抛异常时若冒泡，整个分片进程会退出，
                            #    表现就是"窗口莫名其妙没了"。单只票出错只能算这一只失败。
                            try:
                                r = backfill_one(bs, code, target, miss)
                            except Exception as e:            # noqa: BLE001
                                if _is_session_error(e):
                                    _logout(bs)
                                    bs = _login_retry()
                                    try:
                                        r = backfill_one(bs, code, target, miss)
                                    except Exception as e2:    # noqa: BLE001
                                        r = {"code": code, "status": "error",
                                             "error": f"{type(e2).__name__}: {e2}"}
                                else:
                                    r = {"code": code, "status": "error",
                                         "error": f"{type(e).__name__}: {e}"}
                            logf.write(json.dumps(r, ensure_ascii=False) + "\n")
                            summary[r.get("status", "?")] = summary.get(r.get("status", "?"), 0) + 1
                            time.sleep(random.uniform(0.5 * sleep_s, 1.5 * sleep_s))
                            if r.get("status") != "backfilled" or backfill_only:
                                # backfill_only：只补字段、不追加新交易日
                                #   （迁移场景用：数据边界保持和底稿一致，零行为变化）
                                continue
                        if backfill_only:
                            # 只补字段模式下，字段齐了就算完成 —— 不碰日期维度
                            r = {"code": code, "status": "skipped", "rows": 0}
                            logf.write(json.dumps(r, ensure_ascii=False) + "\n")
                            summary["skipped"] = summary.get("skipped", 0) + 1
                            continue
                        try:
                            existing = pd.read_parquet(target, columns=["date"])
                            # ⚠️ 和【市场的最后交易日】比，不和日历今天比：
                            #    end 默认是今天，周末/节假日永远大于任何数据日期
                            #    ⇒ 用日历比会导致一个都 skip 不掉，把已完成的全部重下 ✗
                            if not existing.empty and str(pd.Timestamp(existing["date"].max()).date()) >= eff_end:
                                r = {"code": code, "status": "skipped", "rows": len(existing)}
                                logf.write(json.dumps(r, ensure_ascii=False) + "\n")
                                summary["skipped"] = summary.get("skipped", 0) + 1
                                continue
                        except Exception:
                            pass  # 文件损坏则重下
                        # 字段已齐、只是日期旧 ⇒ 只追加新交易日（不整段重下 11 年）
                        #   注意：必须走 update_one 而不是 process_one
                        #   —— process_one 是"整段替换"，为补 1 天重下 2800 天的数据
                        try:
                            r = update_one(bs, code, start, end, target)
                        except Exception as e:               # noqa: BLE001
                            r = {"code": code, "status": "error",
                                 "error": f"{type(e).__name__}: {e}"}
                    else:
                        try:
                            r = process_one(bs, code, start, end, target)
                        except Exception as e:               # noqa: BLE001
                            r = {"code": code, "status": "error",
                                 "error": f"{type(e).__name__}: {e}"}
                    if r.get("status") == "error" and _is_session_error(r.get("error", "")):
                        # 会话被服务端踢下线(10001001): 重新登录后重试该只一次
                        _logout(bs)
                        bs = _login_retry()
                        r = process_one(bs, code, start, end, target)
                logf.write(json.dumps(r, ensure_ascii=False) + "\n")
                summary[r["status"]] = summary.get(r["status"], 0) + 1
                # 自动冷却: 连续错误 >=3 时指数退避(5s→60s封顶), 让网络/服务器喘息
                if r.get("status") == "error":
                    consec_errors += 1
                    if consec_errors >= 3:
                        cool = min(5.0 * (2 ** (consec_errors - 3)), 60.0)
                        print(f"[{label}] 连续 {consec_errors} 次错误, 冷却 {cool:.0f}s...", flush=True)
                        time.sleep(cool)
                else:
                    consec_errors = 0
                time.sleep(random.uniform(0.5 * sleep_s, 1.5 * sleep_s))  # 风控友好间隔
    finally:
        _logout(bs)
    return summary


# ============================================================================
# 清单与主流程
# ============================================================================

def get_stock_list(max_attempts: int = 5) -> List[str]:
    """type=='1' 的全部股票(上市中 + 已退市), 按代码排序。
    带重试: 网络不稳时 query_stock_basic 会偶发 10002007(网络接收错误), 一次失败就退出太脆。
    """
    bs = _login_retry()
    try:
        last_error: str = ""
        for i in range(max_attempts):
            rs = bs.query_stock_basic()
            if rs.error_code == "0":
                codes: List[str] = []
                while rs.next():
                    row = rs.get_row_data()
                    d = dict(zip(rs.fields, row))
                    if d.get("type") == "1":
                        codes.append(d["code"])
                return sorted(set(codes))
            last_error = f"{rs.error_code} {rs.error_msg}"
            print(f"[警告] query_stock_basic 失败({last_error}), "
                  f"重试 {i + 1}/{max_attempts}", flush=True)
            time.sleep(min(2 ** i, 10.0))   # 1,2,4,8,10s 退避
        raise RuntimeError(f"query_stock_basic error: {last_error}")
    finally:
        _logout(bs)


def summarize_from_logs(log_paths: List[Path]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for lp in log_paths:
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("status", "?")
            summary[s] = summary.get(s, 0) + 1
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="A股全量(含退市)日线下载器(baostock官方因子)")
    ap.add_argument("--start", default=DEFAULT_PARAMS["start_date"], help="行情起始日 YYYY-MM-DD")
    ap.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="行情结束日 YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=DEFAULT_PARAMS["workers"],
                    help="并发进程数(官方禁止并发, 恒强制为 1, 此项仅供兼容)")
    ap.add_argument("--sleep", type=float, default=DEFAULT_PARAMS["sleep"],
                    help="单只请求间隔秒(±50%%抖动), 风控后建议>=0.5")
    ap.add_argument("--codes", default="", help="逗号分隔指定代码(调试用单进程); 空=全量")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只(全量模式调试用)")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 只(配合 --limit 分片)")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有文件强制重下")
    ap.add_argument("--backfill-only", action="store_true",
                    help="只补缺失字段(如 turn)，不追加新交易日；配合已有文件做字段迁移")
    ap.add_argument("--update", action="store_true",
                    help="增量模式: 只追加新交易日(不重下历史); 文件缺失时自动全量补")
    ap.add_argument("--child", action="store_true", help="内部参数: 子进程模式(勿手动使用)")
    ap.add_argument("--tag", default="", help="内部参数: 日志时间戳统一标签(父子进程共用)")
    args = ap.parse_args()

    # 【合规护栏 2026-09-07】baostock 官方禁止并发连接访问 → 强制单进程串行
    if args.workers != 1:
        print(f"[合规护栏] baostock 官方禁止并发连接, workers 强制为 1 (输入 {args.workers})", flush=True)
        args.workers = 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- 子进程模式: 处理父进程分发的 codes ----------
    if args.child:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        label = f"child-{args.offset}"
        lp = LOG_DIR / f"stock_download_{ts}_child{args.offset}.jsonl"
        print(f"[{label}] 开始处理 {len(codes)} 只", flush=True)
        summary = run_codes(codes, args.start, args.end, not args.no_resume, lp, label,
                            args.sleep, update=args.update, backfill_only=args.backfill_only)
        print(f"[{label}] 完成: {summary}", flush=True)
        return 2 if summary.get("error", 0) else 0

    # ---------- 调试: 指定代码 -> 单进程直接跑 ----------
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        lp = LOG_DIR / f"stock_download_{ts}_direct.jsonl"
        print(f"[direct] 处理 {len(codes)} 只 (start={args.start}, end={args.end})", flush=True)
        summary = run_codes(codes, args.start, args.end, not args.no_resume, lp, "direct",
                            args.sleep, update=args.update, backfill_only=args.backfill_only)
        print(f"[direct] 完成: {summary}", flush=True)
        return 2 if summary.get("error", 0) else 0

    # ---------- 全量: 单进程串行 ----------
    # 官方禁止并发 → workers 恒 1; 直接在父进程内串行跑。
    # 不走 subprocess: 5553 只代码若塞进命令行会超过 Windows 32767 上限, 子进程无法启动。
    print("[1/3] 获取全A股票清单...", flush=True)
    codes = get_stock_list()
    codes = codes[args.offset:]
    if args.limit > 0:
        codes = codes[: args.limit]
    print(f"[1/3] 本次任务 {len(codes)} 只 (start={args.start}, end={args.end}, "
          f"模式={'增量update' if args.update else '全量回补'}, resume={not args.no_resume})", flush=True)

    # 续传提示: 默认 end=今天, 若今天尚未收盘/baostock 未出当日K线,
    # 已下载文件会被判"未达标"而重下 → 应显式指定最近已收盘交易日。
    if (not args.no_resume and args.end >= datetime.today().strftime("%Y-%m-%d")
            and STOCK_DAILY_DIR.exists() and any(STOCK_DAILY_DIR.glob("*.parquet"))):
        print("[提示] 检测到已有 parquet。若下方进度没有大量 skipped, 请用 --end 指定最近已收盘交易日, "
              "例如: --end 2026-09-07", flush=True)

    if not codes:
        print("[1/3] 无代码可处理", flush=True)
        return 0

    lp = LOG_DIR / f"stock_download_{ts}_serial.jsonl"
    print("[2/3] 开始串行下载(可 Ctrl+C 中断, 重跑同命令续传)...", flush=True)
    t0 = time.time()
    summary = run_codes(codes, args.start, args.end, not args.no_resume, lp, "serial",
                        args.sleep, update=args.update, backfill_only=args.backfill_only)
    print(f"[3/3] 完成, 耗时 {(time.time()-t0)/60:.1f} 分钟. 汇总: {summary}", flush=True)
    errs = summary.get("error", 0)
    if errs:
        print(f"      失败 {errs} 只, 重跑同命令自动续传", flush=True)

    meta = {
        "last_update": datetime.now().astimezone().isoformat(),
        "source": "baostock",
        "frequency": "d",
        "start_date": args.start,
        "end_date": args.end,
        "price_type": "raw + official_back_adjust_factor",
        "note": "后复权价 = close * back_adj_factor; 因子为 baostock 官方比例复权因子",
        "mode": "update" if args.update else "full",
        "includes_delisted": True,
        "stock_count": len(codes),
        "summary": summary,
        "worker_logs": [str(lp)],
    }
    STOCK_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STOCK_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元信息写入: {STOCK_META_FILE}", flush=True)
    return 2 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
