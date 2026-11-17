# -*- coding: utf-8 -*-
"""
A股财务（季频）下载器 —— baostock 官方接口，清单驱动 + 断点续传 + 每日预算
========================================================================

设计要点:
  1. 清单驱动：以 data/storage/stock_daily 里的 5477 个代码为清单，逐只跑；
     一只是最小分片单位（47 次请求要么全成、要么下次续）
  2. 断点续传：进度记在 data/storage/fundamental_state.json
     - 已完成的季度按 (year, quarter) 记账，重跑只补缺的季度，不浪费请求
     - 失败的票记 attempts / last_error，下次优先重试
  3. 每日预算：baostock 官方上限 5 万次/天、禁止并发。本器单进程串行，
     按【日期】记账已用请求数（含重试），到 --daily-budget 就保存退出，次日自动续
  4. 落盘：data/storage/fundamental/{table}/{code}.parquet（每股一文件，与 stock_daily 同构）
  5. 每日日志：data/logs/fundamental_YYYYmmdd_HHMMSS.jsonl

用法(示例):
  python data/fetcher/baostock_fundamental_fetcher.py --status
  python data/fetcher/baostock_fundamental_fetcher.py --table profit --priority index
  python data/fetcher/baostock_fundamental_fetcher.py --table profit --codes sh.600000
  python data/fetcher/baostock_fundamental_fetcher.py --table profit --daily-budget 1000 --limit 3

输出:
  data/storage/fundamental/profit/sh.600000.parquet
  data/storage/fundamental_state.json
  data/logs/fundamental_YYYYmmdd_HHMMSS.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.constants import (
    FUNDAMENTAL_DAILY_BUDGET,
    FUNDAMENTAL_DIR,
    FUNDAMENTAL_SLEEP,
    FUNDAMENTAL_STATE_FILE,
    LOG_DIR,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    STOCK_DAILY_DIR,
)

# 表名 → baostock 函数名（运行时 getattr，方便单测注入假对象）
TABLES: Dict[str, str] = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "operation": "query_operation_data",
    "cashflow": "query_cash_flow_data",
}

# 报告期披露缓冲：季度结束后至少 N 天才去问（避免问一个还没披露的期）
DISCLOSE_LAG_DAYS = 30


# ────────────────────────── 纯函数（可单测） ──────────────────────────

def quarter_grid(since_year: int, today: date | None = None,
                 lag_days: int = DISCLOSE_LAG_DAYS) -> List[Tuple[int, int]]:
    """返回 [(year, quarter), ...]，已按时间升序，且跳过"还没到披露窗口"的季度。"""
    today = today or date.today()
    out: List[Tuple[int, int]] = []
    for y in range(since_year, today.year + 1):
        for q, (mm, dd) in enumerate(((3, 31), (6, 30), (9, 30), (12, 31)), start=1):
            end = date(y, mm, dd)
            if (today - end).days >= lag_days:
                out.append((y, q))
    return out


def missing_quarters(done: List[List[int]], grid: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """网格里还没完成的季度（保持升序）。done 是状态文件里的 [[y,q],...]。"""
    have = {(int(y), int(q)) for y, q in done}
    return [t for t in grid if t not in have]


def budget_left(used: int, budget: int) -> int:
    """今天还能发多少次请求。"""
    return max(0, budget - used)


def order_codes(codes: List[str], priority: str, index_codes: List[str]) -> List[str]:
    """index 优先成分股在前；其余按代码升序。两者都保持确定性。"""
    codes = sorted(set(codes))
    if priority != "index":
        return codes
    head = [c for c in codes if c in set(index_codes)]
    tail = [c for c in codes if c not in set(index_codes)]
    return head + tail


# ────────────────────────── 清单与状态 ──────────────────────────

def all_codes() -> List[str]:
    """从 stock_daily 的文件名取代码清单（股名即 sh.600000.parquet）。"""
    return sorted(p.stem for p in STOCK_DAILY_DIR.glob("*.parquet"))


def index_priority_codes() -> List[str]:
    """最新一期指数快照里的 hs300 ∪ zz500 成分。"""
    src = FUNDAMENTAL_DIR.parent / "index_snapshot"
    files = sorted(src.glob("*.parquet"))
    files = [f for f in files if f.stem[:4].isdigit()]
    if not files:
        return []
    snap = pd.read_parquet(files[-1])
    if not {"code", "hs300", "zz500"} <= set(snap.columns):
        return []
    m = snap["hs300"].astype(bool) | snap["zz500"].astype(bool)
    return sorted(snap.loc[m, "code"].astype(str))


def load_state() -> Dict[str, Any]:
    if FUNDAMENTAL_STATE_FILE.exists():
        return json.loads(FUNDAMENTAL_STATE_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "tables": {}, "daily": {}}


def save_state(state: Dict[str, Any]) -> None:
    FUNDAMENTAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    FUNDAMENTAL_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def today_used(state: Dict[str, Any]) -> int:
    return int(state["daily"].get(str(date.today()), 0))


def add_used(state: Dict[str, Any], n: int = 1) -> None:
    k = str(date.today())
    state["daily"][k] = int(state["daily"].get(k, 0)) + n


def code_state(state: Dict[str, Any], table: str, code: str) -> Dict[str, Any]:
    return (state["tables"].setdefault(table, {}).setdefault(
        code, {"done": [], "st": "pending", "att": 0, "err": ""}))


# ────────────────────────── 下载 ──────────────────────────

class Fetcher:
    def __init__(self, state: Dict[str, Any], logf, budget: int):
        self.state = state
        self.logf = logf
        self.budget = budget
        self.bs = None
        self.calls = 0
        self.relogins = 0

    def login(self):
        import baostock as bs
        self.bs = bs
        r = bs.login()
        if r.error_code != "0":
            raise RuntimeError("baostock login 失败: %s" % r.error_msg)

    def logout(self):
        try:
            if self.bs is not None:
                self.bs.logout()
        except Exception:
            pass

    def exhausted(self) -> bool:
        return budget_left(today_used(self.state), self.budget) <= 0

    def relogin(self):
        """会话掉了就重登。baostock 会在长时间运行/休眠/其它进程 logout 后踢掉会话。"""
        try:
            self.bs.logout()
        except Exception:
            pass
        time.sleep(1.0)
        self.login()
        self.relogins += 1

    def _call(self, fn, **kw):
        """一次 API 调用：计数 + 限速 + 预算检查 + 会话失效自动重连。"""
        if self.exhausted():
            return None
        # 主动换会话：长跑里会话平均 1~2 千次后开始不稳
        if self.calls and self.calls % 1500 == 0:
            self.relogin()
        last = None
        for attempt in range(3):
            add_used(self.state, 1)
            self.calls += 1
            time.sleep(FUNDAMENTAL_SLEEP * random.uniform(0.5, 1.5))
            try:
                rs = fn(**kw)
            except Exception as e:                      # socket 断了
                last = "socket: %s" % e
                self.relogin()
                continue
            if rs.error_code == "0":
                rows = []
                while rs.next():
                    rows.append(dict(zip(rs.fields, rs.get_row_data())))
                return rows
            last = "%s: %s" % (rs.error_code, rs.error_msg)
            if str(rs.error_code).startswith("10001"):   # 未登录 / 会话失效
                self.relogin()
                continue
            if str(rs.error_code).startswith("10004"):   # 参数类错误，重试无益
                raise RuntimeError(last)
            time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        raise RuntimeError("重试 3 次仍失败：%s" % last)

    def fetch_code(self, table: str, code: str, grid, done) -> Tuple[str, int]:
        """拉一只票缺的季度，落盘。返回 (状态, 本次请求数)。"""
        fn = getattr(self.bs, TABLES[table])
        target = FUNDAMENTAL_DIR / table / (code + ".parquet")
        old = pd.read_parquet(target) if target.exists() else pd.DataFrame()
        # 以【落盘文件】为准：状态文件可能被删/换机器，但 parquet 里的 statDate 不会骗人
        if not done and not old.empty and "statDate" in old.columns:
            done = [[int(sd[:4]), (int(sd[5:7]) + 2) // 3]
                    for sd in old["statDate"].astype(str)]
        need = missing_quarters(done, grid)
        if not need:
            return "done", 0
        new_rows: List[Dict[str, Any]] = []
        n = 0
        for (y, q) in need:
            rows = self._call(fn, code=code, year=y, quarter=q)
            if rows is None:                      # 预算用完，提前返回
                return "budget", n
            n += 1
            new_rows.extend(rows)
        df = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
        if not df.empty and "statDate" in df.columns:
            df = df.drop_duplicates(subset=["statDate"], keep="last").sort_values("statDate")
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)
        done.extend([[y, q] for (y, q) in need])
        return "done", n


def status_report(state: Dict[str, Any]) -> None:
    print("  表        已完成  部分  待跑  失败   %d 天已用请求" % len(state["daily"]))
    for table in TABLES:
        cs = state["tables"].get(table, {})
        done = sum(1 for v in cs.values() if v["st"] == "done")
        part = sum(1 for v in cs.values() if v["st"] == "partial")
        fail = sum(1 for v in cs.values() if v["st"] == "failed")
        pend = len(all_codes()) - done - part - fail
        print("  %-9s %6d %5d %5d %5d" % (table, done, part, pend, max(fail, 0)))
    print("  今天已用：%d 次（预算 %d）" % (today_used(state), FUNDAMENTAL_DAILY_BUDGET))


def main() -> None:
    ap = argparse.ArgumentParser(description="A股财务季频下载器（baostock）")
    ap.add_argument("--tables", default=",".join(TABLES),
                    help="逗号分隔，默认全部 5 张：profit,growth,balance,operation,cashflow")
    ap.add_argument("--since", type=int, default=2015, help="起始年份")
    ap.add_argument("--priority", default="index", choices=["index", "code"])
    ap.add_argument("--codes", default="", help="逗号分隔，手动指定一批（调试用）")
    ap.add_argument("--daily-budget", type=int, default=FUNDAMENTAL_DAILY_BUDGET)
    ap.add_argument("--limit", type=int, default=0, help="本次最多处理多少只（0=不限）")
    ap.add_argument("--retry-failed-first", action="store_true", default=True)
    ap.add_argument("--status", action="store_true", help="只看进度，不下数据")
    args = ap.parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise SystemExit("未知表: %r，可选 %s" % (bad, sorted(TABLES)))

    state = load_state()

    if args.status:
        status_report(state)
        return

    # 清单
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        codes = all_codes()
        idx = index_priority_codes() if args.priority == "index" else []
        codes = order_codes(codes, args.priority, idx)
    if not codes:
        raise SystemExit("清单为空：检查 data/storage/stock_daily")

    grid = quarter_grid(args.since)
    per_stock = len(tables) * len(grid)
    per_day = max(1, args.daily_budget // per_stock)
    print("  表 %s（%d 张）｜ 清单 %d 只 ｜ 季度 %d 个（%s ~ %s）｜ 预算 %d"
          % (",".join(tables), len(tables), len(codes), len(grid),
             "%dQ%d" % grid[0], "%dQ%d" % grid[-1], args.daily_budget))
    print("  每只票 %d 次请求 ⇒ 每天约 %d 只 ⇒ 全量约 %.1f 天"
          % (per_stock, per_day, len(codes) / per_day))

    print("  提示：每只票约 %d 次请求、约 %.0f 秒；每只都会打一行，不会卡住"
          % (per_stock, per_stock * 0.31), flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logpath = LOG_DIR / ("fundamental_%s.jsonl" % datetime.now().strftime("%Y%m%d_%H%M%S"))

    t_start = time.time()
    f = Fetcher(state, logpath.open("a", encoding="utf-8"), args.daily_budget)
    f.login()
    done_n = 0
    try:
        for code in codes:
            # 一只票：把所有表都补齐，才算出这只票完成
            todo = []
            for tb in tables:
                cs = code_state(state, tb, code)
                if cs["st"] == "done" and not missing_quarters(cs["done"], grid):
                    continue
                todo.append((tb, cs))
            if not todo:
                continue
            try:
                st, n = "done", 0
                for tb, cs in todo:
                    st1, n1 = f.fetch_code(tb, code, grid, cs["done"])
                    n += n1
                    if st1 == "budget":
                        st = "budget"
                        cs["st"] = "partial" if cs["done"] else "pending"
                        break
                    cs["st"] = "done" if not missing_quarters(cs["done"], grid) else "partial"
                    cs["err"] = ""
                cs = todo[0][1]
            except Exception as e:
                cs["att"] = int(cs.get("att", 0)) + 1
                cs["err"] = "%s: %s" % (type(e).__name__, e)
                cs["st"] = "failed"
                f.logf.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"),
                                         "code": code, "err": cs["err"]}, ensure_ascii=False) + "\n")
                f.logf.flush()
                save_state(state)
                if cs["att"] >= MAX_RETRIES:
                    continue
                time.sleep(RETRY_BACKOFF_BASE * (2 ** cs["att"]))
                continue
            if st == "budget":
                cs["st"] = "partial" if cs["done"] else "pending"
                save_state(state)
                print("  今日预算用完：已处理 %d 只，明天接着跑" % done_n)
                break
            cs["st"] = "done" if not missing_quarters(cs["done"], grid) else "partial"
            cs["err"] = ""
            done_n += 1
            save_state(state)
            _el = time.time() - t_start
            _used = today_used(state)
            _left_req = max(0, args.daily_budget - _used)
            _eta_h = _left_req * 0.31 / 3600
            print("  [%3d] %-11s ✓ 本次 %3d 次 ｜ 今日 %5d/%d ｜ 已用 %5.1f 分 ｜ 今日剩余约 %.1f 时"
                  % (done_n, code, n, _used, args.daily_budget, _el / 60, _eta_h),
                  flush=True)
            if args.limit and done_n >= args.limit:
                print("  达到 --limit %d，停止" % args.limit)
                break
    finally:
        f.logout()
        save_state(state)
        try:
            f.logf.close()
        except Exception:
            pass

    print("  本次完成 %d 只 ｜ 今日累计请求 %d ｜ 重连 %d 次 ｜ 日志 %s"
          % (done_n, today_used(state), f.relogins, logpath.name))
    print("  下次直接重跑同一条命令即可（已完成的票自动跳过）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  已中断（进度已保存，重跑自动续）")
        sys.exit(130)
