"""tests/test_fetcher.py —— 补字段能力的离线回归测试

跑法（项目根目录）：
    python tests/test_fetcher.py

特点：
    · 不联网、不碰 data/storage、< 1 秒
    · 卡在【baostock 客户端边界】上（FakeBS），不 monkeypatch 内部函数
      ⇒ fetch_kline 内部重构时测试不会假红
    · 每条断言都对应 2026-11-09 实际踩过的坑（见各节标题）
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.fetcher.baostock_stock_fetcher as F  # noqa: E402

CHECKS = []


def check(cond, msg):
    CHECKS.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


# ============================================================================
# FakeBS：只实现 baostock 的公开查询接口
#   ⇒ 真实的 fetch_kline / backfill_one 都在测试里跑一遍，测的是行为不是实现
# ============================================================================

class FakeRS:
    def __init__(self, fields, rows, error_code="0", error_msg="success"):
        self.fields = fields.split(",")
        self.error_code = error_code
        self.error_msg = error_msg
        self._rows = rows
        self._i = -1

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


class FakeBS:
    """table: {code: DataFrame(date, close, volume, turn, ...)}"""

    def __init__(self, table):
        self.table = table

    def query_history_k_data_plus(self, code, fields, start_date, end_date,
                                  frequency="d", adjustflag="3"):
        want = fields.split(",")
        df = self.table.get(code)
        if df is None:
            return FakeRS(fields, [])
        d = df[(df["date"] >= pd.Timestamp(start_date))
               & (df["date"] <= pd.Timestamp(end_date))]
        rows = []
        for _, r in d.iterrows():
            row = []
            for c in want:
                if c == "date":
                    row.append(str(r["date"].date()))
                elif c == "code":
                    row.append(code)
                else:
                    v = r.get(c, np.nan)
                    row.append("" if pd.isna(v) else str(v))
            rows.append(row)
        return FakeRS(fields, rows)


def _write(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ============================================================================
# 坑 ②：_is_session_error 只吃字符串 ⇒ 传异常对象时 TypeError ⇒ 分片进程被杀
# ============================================================================
print("\n\u2500\u2500 ① _is_session_error\u3010\u5751\uff1a\u5f02\u5e38\u5bf9\u8c61\u4e0d\u53ef iterable\u3011\u2500\u2500")

check(F._is_session_error("10001001: \u7528\u6237\u672a\u767b\u5f55"), "字符串含错误码 \u21d2 True")
check(F._is_session_error("\u7528\u6237\u672a\u767b\u5f55"), "字符串含中文 \u21d2 True")
check(F._is_session_error(RuntimeError("10001001: \u7528\u6237\u672a\u767b\u5f55")),
      "\u5f02\u5e38\u5bf9\u8c61\u542b\u9519\u8bef\u7801 \u21d2 True\uff08\u6628\u5929\u5c31\u662f\u8fd9\u91cc\u629b TypeError\uff09")
check(F._is_session_error(RuntimeError("timed out")) is False, "\u5f02\u5e38\u5bf9\u8c61\u4e0d\u542b\u7801 \u21d2 False")
check(F._is_session_error("") is False, "\u7a7a\u5b57\u7b26\u4e32 \u21d2 False")
check(F._is_session_error("no_data_in_window") is False, "\u666e\u901a\u72b6\u6001\u5b57\u7b26\u4e32 \u21d2 False")


# ============================================================================
# 坑 ③：整列全 NaN 不写盘 ⇒ 洞永远补不完；object 型数值列也算缺
# ============================================================================
print("\n\u2500\u2500 ② missing_fields\u3010\u5751\uff1aobject \u578b\u4e0d\u7b97\u7f3a\u3001\u5168 NaN \u4e0d\u5199\u3011\u2500\u2500")

_REQ = list(F._REQUIRED_FIELDS)

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    base = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"])})
    for c in ("open", "high", "low", "close", "volume", "amount"):
        base[c] = 1.0
    base["tradestatus"] = 1
    base["isST"] = 0
    base["back_adj_factor"] = 1.0
    base["code"] = "sh.600000"

    p_ok = tmp / "ok.parquet"
    ok_df = base.copy()
    ok_df["turn"] = [0.5, 0.6]
    _write(p_ok, ok_df)
    check(F.missing_fields(p_ok) == [], "12 \u5217\u9f50\u4e14 turn \u662f float \u21d2 \u7f3a = []")

    p_no = tmp / "noturn.parquet"
    _write(p_no, base)
    check(F.missing_fields(p_no) == ["turn"], "\u7f3a turn \u5217 \u21d2 \u7f3a = ['turn']")

    p_obj = tmp / "obj.parquet"
    obj_df = base.copy()
    obj_df["turn"] = ["0.5", "0.6"]   # object\uff08\u5b57\u7b26\u4e32\uff09
    _write(p_obj, obj_df)
    check(F.missing_fields(p_obj) == ["turn"],
          "turn \u5b58\u5728\u4f46\u662f object \u578b \u21d2 \u4ecd\u5224\u4e3a\u7f3a\uff08\u5426\u5219 update \u65f6 ArrowTypeError\uff09")

    check(F.missing_fields(tmp / "nope.parquet") == _REQ, "\u6587\u4ef6\u4e0d\u5b58\u5728 \u21d2 \u5168\u90e8\u5b57\u6bb5\u90fd\u7f3a")

    p_bad = tmp / "bad.parquet"
    p_bad.write_bytes(b"not a parquet")
    check(F.missing_fields(p_bad) == _REQ, "\u574f parquet \u21d2 \u5f53\u4f5c\u5168\u7f3a\uff08\u4e0d\u629b\u5f02\u5e38\uff09")


# ============================================================================
# 坑 ④：backfill 的三个必须——close 对齐、date 不撞名、全 NaN 也要写
# ============================================================================
print("\n\u2500\u2500 ③ backfill_one\u3010\u5751\uff1aclose \u4e0d\u4e00\u81f4\u3001date \u649e\u540d\u3001\u5168 NaN \u4e0d\u5199\u3011\u2500\u2500")

def _mk_base(n=3, close=(10.0, 10.5, 11.0)):
    d = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"][:n])})
    d["code"] = "sh.600000"
    d["open"] = list(close)
    d["high"] = list(close)
    d["low"] = list(close)
    d["close"] = list(close)
    d["volume"] = [1000, 1100, 1200][:n]
    d["amount"] = [1e4, 1.1e4, 1.2e4][:n]
    d["tradestatus"] = 1
    d["isST"] = 0
    d["back_adj_factor"] = 1.0
    return d

COLS_UNCHANGED = ["open", "high", "low", "close", "volume", "amount",
                  "tradestatus", "isST", "back_adj_factor"]

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    old = _mk_base()

    # --- 正常：补上 turn，其余列必须逐位不变 ---
    p = tmp / "sh.600000.parquet"
    _write(p, old)
    feed = old.copy(); feed["turn"] = [0.5, 0.55, 0.6]
    r = F.backfill_one(FakeBS({"sh.600000": feed}), "sh.600000", p, ["turn"])
    got = pd.read_parquet(p)
    check(r["status"] == "backfilled", "\u6b63\u5e38\u8865\u5b57\u6bb5 \u21d2 status=backfilled")
    check(len(got.columns) == len(old.columns) + 1, "\u5217\u6570 +1\uff08old %d \u21d2 %d\uff09" % (len(old.columns), len(got.columns)))
    check(str(got["turn"].dtype) == "float64", "turn \u843d\u76d8\u662f float64\uff08\u4e0d\u662f object\uff09")
    same = all(np.allclose(got[c].to_numpy(dtype=float), old[c].to_numpy(dtype=float),
                           rtol=0, atol=0) for c in COLS_UNCHANGED)
    check(same, "\u5176\u4f59 9 \u5217\u9010\u4f4d\u4e0d\u53d8\uff08\u8fd9\u662f\u672c\u6b21\u6539\u52a8\u7684\u6838\u5fc3\u627f\u8bfa\uff09")
    check(list(got.columns).count("date") == 1, "date \u5217\u6ca1\u88ab merge \u649e\u6210 date_x/date_y")

    # --- close 不一致：必须抛，且文件不能被写 ---
    p2 = tmp / "bad.parquet"
    _write(p2, old)
    before = pd.read_parquet(p2)
    feed2 = old.copy(); feed2["close"] = feed2["close"] * 4.8   # \u6a21\u62df\u9001\u8f6c\u540e\u91cd\u7b97
    feed2["turn"] = 0.5
    raised = False
    try:
        F.backfill_one(FakeBS({"sh.600000": feed2}), "sh.600000", p2, ["turn"])
    except RuntimeError as e:
        raised = "\u5bf9\u9f50\u5931\u8d25" in str(e)
    check(raised, "close \u4e0d\u4e00\u81f4 \u21d2 \u629b RuntimeError\uff08\u7edd\u4e0d\u9759\u9ed8\u5199\uff09")
    check("turn" not in pd.read_parquet(p2).columns, "\u629b\u5f02\u5e38\u540e\u6587\u4ef6\u672a\u88ab\u4fee\u6539")
    check(list(pd.read_parquet(p2).columns) == list(before.columns), "\u5217\u96c6\u5408\u672a\u53d8")

    # --- turn 全空（全程停牌）：仍要写列，状态 backfilled_nan ---
    p3 = tmp / "allnan.parquet"
    _write(p3, old)
    feed3 = old.copy(); feed3["turn"] = np.nan
    r3 = F.backfill_one(FakeBS({"sh.600000": feed3}), "sh.600000", p3, ["turn"])
    got3 = pd.read_parquet(p3)
    check(r3["status"] == "backfilled_nan", "turn \u5168\u7a7a \u21d2 status=backfilled_nan\uff08\u4e0d\u662f\u5931\u8d25\uff09")
    check("turn" in got3.columns and got3["turn"].isna().all(),
          "\u5168 NaN \u4e5f\u8981\u5199\u5217\uff08\u5426\u5219\u4ee5\u540e\u6bcf\u6b21\u90fd\u5f53\u6d1e\u91cd\u8bd5\uff0c\u6c38\u8fdc\u8865\u4e0d\u5b8c\uff09")
    check(str(got3["turn"].dtype) == "float64", "\u5168 NaN \u7684 turn \u4e5f\u8981\u662f float64")


# ============================================================================
# 原有逻辑：_merge_new_rows（一直没测过，补上保护）
# ============================================================================
print("\n\u2500\u2500 \u2463 _merge_new_rows\u3010\u539f\u6709\u903b\u8f91\uff0c\u8865\u4fdd\u62a4\u3011\u2500\u2500")

a = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "v": [1, 2]})
b = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-06"]), "v": [99, 3]})
m = F._merge_new_rows(a, b)
check(len(m) == 3, "\u540c\u65e5\u53bb\u91cd\uff1a2 + 2 \u884c \u21d2 3 \u884c")
check(m["v"].tolist() == [1, 99, 3], "\u540c\u65e5\u4fdd\u7559\u65b0\u503c\uff081/3 \u90a3\u5929\u53d6 99\uff09")
check(m["date"].is_monotonic_increasing, "\u7ed3\u679c\u6309\u65e5\u671f\u5347\u5e8f")


# ============================================================================
# 坑 ①：resume 拿【日历今天】比较 ⇒ 周末一个都 skip 不掉 ⇒ 已完成的全部重下
# ============================================================================
print("\n\u2500\u2500 \u2464 run_codes \u7684 resume \u5224\u5b9a\u3010\u5751\uff1a\u548c\u65e5\u5386\u4eca\u5929\u6bd4\u3011\u2500\u2500")

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    stock_dir = tmp / "stock_daily"
    stock_dir.mkdir(parents=True)
    cls = _mk_base()
    cls["turn"] = [0.5, 0.55, 0.6]
    _write(stock_dir / "sh.600000.parquet", cls)
    _write(stock_dir / "sh.600001.parquet", cls)   # \u4e24\u4e2a\u6587\u4ef6\uff0c\u6700\u672b\u65e5\u90fd\u662f 2020-01-06

    old_dir = F.STOCK_DAILY_DIR
    F.STOCK_DAILY_DIR = stock_dir
    # 把登录也替掉：run_codes 开头会 _login_retry()，不替就变成联网测试
    old_login, old_logout = F._login_retry, F._logout
    F._login_retry = lambda: FakeBS({})
    F._logout = lambda _bs=None: None
    try:
        log = tmp / "x.jsonl"
        # end \u7528\u4e00\u4e2a\u3010\u665a\u4e8e\u6240\u6709\u6570\u636e\u3011\u7684\u65e5\u671f\uff0c\u6a21\u62df\u201c\u5468\u672b\u8dd1\u201d
        summ = F.run_codes(["sh.600000", "sh.600001"], "2019-01-01", "2030-12-31",
                            True, log, "t", 0.0, update=False, backfill_only=False)
        check(summ.get("skipped", 0) == 2,
              "\u5b57\u6bb5\u9f50\u4e14\u5df2\u5230\u5e02\u573a\u6700\u672b\u65e5 \u21d2 \u4e24\u53ea\u90fd skipped\uff08\u5b9e\u9645 %r\uff09" % (summ,))
        check("ok" not in summ, "\u4e0d\u8be5\u51fa\u73b0 ok\uff08\u5426\u5219\u5c31\u662f\u62ff\u65e5\u5386\u4eca\u5929\u53bb\u6bd4\uff09")
    finally:
        F.STOCK_DAILY_DIR = old_dir
        F._login_retry, F._logout = old_login, old_logout


# ============================================================================
passed = sum(1 for c in CHECKS if c)
total = len(CHECKS)
print("\n" + "=" * 46)
print("  结果: 通过 %d / %d" % (passed, total))
if passed != total:
    print("  仍有 %d 项失败 \u274c 请按上面提示修复" % (total - passed))
