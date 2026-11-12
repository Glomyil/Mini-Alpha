"""tests/test_pool.py —— 样本空间 Pool 的契约

四个必须钉死的点：
  A. can_trade 是【同一对象】引用（研究口径永远不许改它）
  B. 热身区（研究区间之前）【不施加限制】，且不把超范围的日期喂给 loader
  C. invest_pool ⊆ 全部票，selectable() = can_trade ∩ invest_pool
  D. label 必须同时含两个池名（只印一个等于没说）

B 是这次实测撞出来的：面板从 2015-01-05 起，快照从 2015-01-30 起，
   make_pool 若不裁到研究区间，load_index_mask 会按设计抛
   "目标日早于最早一份快照 —— 拒绝用未来名单回填"。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest.pool as POOL  # noqa: E402

CK = []


def ck(cond, msg):
    CK.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


# fixture：面板 2015-01 ~ 2015-06，研究起点 2015-04 ⇒ 前三个月是热身区
DATES = pd.date_range("2015-01-05", "2016-06-30", freq="B")   # 跨过 2015 整年热身区
COLS = ["sh.600000", "sh.600001", "sz.000001"]
panels = {
    "hfq_close": pd.DataFrame(1.0, index=DATES, columns=COLS),
    "can_trade": pd.DataFrame(True, index=DATES, columns=COLS),
}

_calls = []


# ★ fake 必须让不同宇宙产生不同掩码，否则
#   「estimation_pool & invest_pool」和「只用 invest_pool」永远一样 ⇒ 测不出基准池用错了。
#   设定：hs300 = 前 2 只；zz500 = 前 3 只（包含 hs300）。
_FAKE_KEYS = {"hs300": 2, "zz500": 3, "zz800": 3, "sz50": 1}


def _fake_loader(index_code, target_dates, columns, *, src=None):
    ds = list(target_dates)
    _calls.append({"code": index_code, "n": len(ds),
                   "first": pd.Timestamp(min(ds)), "last": pd.Timestamp(max(ds))})
    k = _FAKE_KEYS.get(index_code, len(columns))
    df = pd.DataFrame(True, index=pd.DatetimeIndex(ds), columns=list(columns), dtype=bool)
    df.iloc[:, k:] = False
    return df


POOL.load_index_mask = _fake_loader            # 只替换本模块里的引用

spec = SimpleNamespace(since=2016, universe_stage1="hs300", universe_stage2="zz500")
pl = POOL.make_pool(panels, spec)

print("\n\u2500\u2500 A. can_trade \u662f\u540c\u4e00\u5bf9\u8c61\u2500\u2500")
ck(pl.can_trade is panels["can_trade"], "pool.can_trade \u5c31\u662f panels['can_trade']\uff08\u540c\u4e00\u5bf9\u8c61\uff0c\u672a\u590d\u5236\uff09")

print("\n\u2500\u2500 B. \u70ed\u8eab\u533a\u4e0d\u65bd\u52a0\u9650\u5236\u2500\u2500")
ck(len(_calls) == 2, f"\u4e24\u4e2a\u6c60\u5404\u8c03\u4e00\u6b21 loader\uff08\u5b9e\u6d4b {len(_calls)}\uff09")
for c in _calls:
    ck(c["first"] >= pd.Timestamp("2016-01-01"),
       f"{c['code']} \u53ea\u628a >= 2016-01-01 \u7684\u65e5\u671f\u5582\u7ed9 loader\uff08\u5b9e\u6d4b\u6700\u65e9 {c['first'].date()}\uff09")
_warm = DATES < pd.Timestamp("2016-01-01")
ck(bool(pl.invest_pool.loc[_warm].to_numpy().all()),
   f"\u70ed\u8eab\u533a\uff08{int(_warm.sum())} \u5929\uff09\u7684 invest_pool \u5fc5\u987b\u5168 True")
ck(bool(pl.estimation_pool.loc[_warm].to_numpy().all()),
   "\u70ed\u8eab\u533a\u7684 estimation_pool \u4e5f\u5168 True")
ck(pl.invest_pool.shape == panels["hfq_close"].shape, f"\u6c60\u5f62\u72b6 == \u9762\u677f\u5f62\u72b6\uff08{pl.invest_pool.shape}\uff09")
ck(pl.invest_pool.to_numpy().dtype == bool, "dtype \u5fc5\u987b\u662f bool")

print("\n\u2500\u2500 C. selectable = can_trade \u2229 invest_pool\u2500\u2500")
sel = pl.selectable()
ck(bool((sel.to_numpy() <= pl.can_trade.to_numpy()).all()), "selectable \u2208 can_trade")
ck(bool((sel.to_numpy() <= pl.invest_pool.to_numpy()).all()), "selectable \u2208 invest_pool")
_m = pd.DataFrame(False, index=DATES, columns=COLS)
_m.loc[DATES[-1], COLS[0]] = True
panels["can_trade"] = panels["can_trade"].copy()
panels["can_trade"].loc[DATES[-1], COLS[0]] = False
pl2 = POOL.make_pool(panels, spec)
ck(not bool(pl2.selectable().loc[DATES[-1], COLS[0]]),
   "invest_pool \u91cc\u6709\u3001\u4f46 can_trade=False \u7684\u7968 \u21d2 selectable \u5fc5\u987b\u4e0d\u542b\u5b83")

print("\n\u2500\u2500 D. label \u2500\u2500")
ck("estimation_pool=hs300" in pl.label and "invest_pool=zz500" in pl.label,
   f"label \u540c\u65f6\u542b\u4e24\u4e2a\u6c60\uff1a{pl.label}")



print("\n── E. benchmark_mask\uff08§4.2）──")
# 基准的样本必须 = 两个点位的交集。
# 只用 invest_pool 会让基准里出现策略根本买不到的票
# （点位1 已把它们的因子值置 NaN）⇒ 超额不可信。
# 实测代价：hs300 那一跑的超额从 −0.92% 变成 +2.92%。
_bm = pl.benchmark_mask()
_est_only = pl.estimation_pool
_inv_only = pl.invest_pool
ck(bool((_bm.to_numpy() == (_est_only.to_numpy() & _inv_only.to_numpy())).all()),
   "benchmark_mask == estimation_pool & invest_pool")

_spec_est = SimpleNamespace(since=2016, universe_stage1="hs300", universe_stage2="all")
pl_est = POOL.make_pool(panels, _spec_est)
ck(bool((pl_est.benchmark_mask().to_numpy() == pl_est.estimation_pool.to_numpy()).all()),
   "est=hs300 / inv=all ⇒ 基准池必须跟着缩到 estimation_pool（不能放到全市场）")

_spec_all = SimpleNamespace(since=2016, universe_stage1="all", universe_stage2="hs300")
pl_all = POOL.make_pool(panels, _spec_all)
ck(bool((pl_all.benchmark_mask().to_numpy() == pl_all.invest_pool.to_numpy()).all()),
   "est=all / inv=hs300 ⇒ 基准池 = invest_pool")

# ★ 这条是“为什么不能只用 invest_pool”的可执行证据：
#   est 不是 all 时，两者不同 ⇒ 若拿 invest_pool 当基准就是错的。
ck(bool((pl_est.invest_pool.to_numpy() != pl_est.benchmark_mask().to_numpy()).any()),
   "est=hs300 时 invest_pool ≠ benchmark_mask（所以不能只用 invest_pool）")

passed = sum(1 for c in CK if c)
print("\n" + "=" * 46)
print("  \u7ed3\u679c: \u901a\u8fc7 %d / %d" % (passed, len(CK)))
if passed != len(CK):
    print("  \u4ecd\u6709 %d \u9879\u5931\u8d25 \u274c" % (len(CK) - passed))
    sys.exit(1)
