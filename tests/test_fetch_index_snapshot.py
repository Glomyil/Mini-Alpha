"""tests/test_fetch_index_snapshot.py —— 抓取脚本的行为单元

Seam 说明（tdd skill 要求）
    测的是 ops.fetch_index_snapshot._build_one(bs, date)。它带下划线，但对脚本而言是
    【行为单元】（fake bs 进、带列的表出），不是实现细节 —— 而「名单生效日不许晚于
    快照日」这条前视防线只能在这里钉。

能抓什么
    · 前视名单混入（生效日 > 快照日 ⇒ 必须报错，不许落盘）
    · zz800 ≠ hs300 ∪ zz500（中证800 是拼出来的，拼错了不会有人发现）
    · 列名 / dtype / code 去重 / 溯源行数
    · 月末交易日选取（与调仓频率解耦）

抓不到什么
    · 真实网络行为、baostock 实际返回内容
    · 断点续跑 / 会话重连（在 main() 里，需要 CLI + 文件系统）
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ops.fetch_index_snapshot as F  # noqa: E402

F.SLEEP_S = 0.0            # 测试里不睡

CK = []


def ck(cond, msg):
    CK.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


class _RS:
    """仿 baostock 的结果集：error_code / fields / next / get_row_data"""

    def __init__(self, rows, fields=("updateDate", "code")):
        self.error_code = "0"
        self.error_msg = ""
        self.fields = list(fields)
        self._rows = rows
        self._i = -1

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


def make_bs(eff, hs300, zz500, sz50):
    """造一个假 baostock：三个查询函数各回一份名单"""

    class _BS:
        pass

    b = _BS()
    for fn, codes in (("query_hs300_stocks", hs300),
                      ("query_zz500_stocks", zz500),
                      ("query_sz50_stocks", sz50)):
        rows = [[eff, c] for c in codes]
        setattr(b, fn, (lambda rows=rows: (lambda date=None: _RS(rows)))())
    return b


print("\n\u2500\u2500 \u2460 zz800 \u5fc5\u987b\u7b49\u4e8e hs300 \u222a zz500\u2500\u2500")
bs = make_bs("2020-06-30", ["sh.600000", "sh.600001"], ["sz.000001", "sz.000002"], ["sh.600000"])
out, prov = F._build_one(bs, "2020-06-30")
ck(set(out.columns) == {"code", "hs300", "zz500", "zz800", "sz50"},
   f"\u5217\u540d = code + 4 \u4e2a\u5e03\u5c14\u9879\uff08\u5b9e\u6d4b {list(out.columns)}\uff09")
ck(bool((out["zz800"] == (out["hs300"] | out["zz500"])).all()),
   "zz800 == hs300 | zz500")
_zz = int(out["zz800"].sum()); _hs = int(out["hs300"].sum())
ck(_zz == 4 and _hs == 2, f"zz800 只数 {_zz}（应 4）、hs300 {_hs}（应 2）")
ck(out["code"].is_unique, "code \u5fc5\u987b\u53bb\u91cd")
ck(out.dtypes.drop("code").eq(bool).all(), "4 \u4e2a\u5e03\u5c14\u5217 dtype \u5fc5\u987b\u662f bool")

print("\n\u2500\u2500 \u2461 \u6eaf\u6e90\uff08\u6bcf\u4e2a\u6307\u6570\u4e00\u884c\uff09\u2500\u2500")
ck(len(prov) == 3, f"\u4e09\u4e2a\u6307\u6570 \u21d2 3 \u884c\u6eaf\u6e90\uff08\u5b9e\u6d4b {len(prov)}\uff09")
ck(all(r["effective_date"] == "2020-06-30" for r in prov), "\u751f\u6548\u65e5\u5fc5\u987b\u539f\u6837\u8bb0\u4e0b\u6765")
ck({r["index_code"] for r in prov} == {"hs300", "zz500", "sz50"}, "\u6eaf\u6e90\u91cc\u8981\u6709\u4e09\u4e2a\u6307\u6570\u540d")

print("\n\u2500\u2500 \u2462 \u2605 \u524d\u89c6\u9632\u7ebf\uff1a\u751f\u6548\u65e5\u665a\u4e8e\u5feb\u7167\u65e5 \u21d2 \u5fc5\u987b\u62a5\u9519\u2500\u2500")
bs_future = make_bs("2020-07-15", ["sh.600000"], ["sz.000001"], [])
try:
    F._build_one(bs_future, "2020-06-30")
    ck(False, "\u751f\u6548\u65e5 2020-07-15 > \u5feb\u7167\u65e5 2020-06-30 \u21d2 \u7adf\u7136\u6ca1\u62a5\u9519\uff08\u524d\u89c6\u4f1a\u88ab\u843d\u76d8\uff09")
except AssertionError as e:
    ck("2020-07-15" in str(e) and "2020-06-30" in str(e),
       f"\u62a5\u9519\u4e14\u6d88\u606f\u91cc\u5e26\u4e24\u4e2a\u65e5\u671f\uff1a{str(e)[:60]}")
except Exception as e:                                    # noqa: BLE001
    ck(False, f"\u62a5\u7684\u662f {type(e).__name__}\uff0c\u5e94\u8be5\u662f AssertionError")

bs_eq = make_bs("2020-06-30", ["sh.600000"], ["sz.000001"], [])
try:
    F._build_one(bs_eq, "2020-06-30")
    ck(True, "\u751f\u6548\u65e5 == \u5feb\u7167\u65e5 \u21d2 \u5141\u8bb8\uff08\u4e0d\u80fd\u8bef\u6740\u540c\u65e5\u751f\u6548\u7684\u540d\u5355\uff09")
except Exception as e:                                    # noqa: BLE001
    ck(False, f"\u540c\u65e5\u751f\u6548\u88ab\u8bef\u6740\uff1a{type(e).__name__}")

print("\n\u2500\u2500 \u2463 \u6708\u672b\u4ea4\u6613\u65e5\u9009\u53d6\uff08\u4e0e\u8c03\u4ed3\u9891\u7387\u89e3\u8026\uff09\u2500\u2500")
ix = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2021-03-31"])
got = F._month_end_trading_days(ix, since=2020)
ck([str(d.date()) for d in got] == ["2020-01-31", "2020-02-28", "2021-03-31"],
   f"\u53ea\u53d6\u6bcf\u4e2a\u81ea\u7136\u6708\u7684\u6700\u540e\u4e00\u4e2a\u4ea4\u6613\u65e5\uff08\u5b9e\u6d4b {[str(d.date()) for d in got]}\uff09")
ck(all(d.year >= 2021 for d in F._month_end_trading_days(ix, since=2021)),
   "since \u8fc7\u6ee4\u751f\u6548\uff08\u53ea\u4fdd\u7559 >= since \u5e74\u7684\uff09")

passed = sum(1 for c in CK if c)
print("\n" + "=" * 46)
print("  \u7ed3\u679c: \u901a\u8fc7 %d / %d" % (passed, len(CK)))
if passed != len(CK):
    print("  \u4ecd\u6709 %d \u9879\u5931\u8d25 \u274c" % (len(CK) - passed))
    sys.exit(1)