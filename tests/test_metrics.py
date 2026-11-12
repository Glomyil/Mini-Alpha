# -*- coding: utf-8 -*-
"""
tests/test_metrics.py —— metrics.py 验收（手算 fixture，不联网、不依赖真面板）

  ① 常数日收益：几何年化手算；波动 ≈ 0；夏普必须是 NaN（不许 inf）
  ② 最大回撤：先涨到 1.1 再跌到 0.88 → −20%；符号为负；单调上涨 → 0
  ③ 成本口径：累计换手 / 累计费用 / 年化换手 = 累计 ÷ 年数
  ④ IC：同序 → +1；反序 → −1；无关 → 手算 0.2 / 0.8；截面不足 min_names → 该期跳过
  ⑤ 三种多空定义：各自手算，且**互不相等**（别写成一个）
  ⑥ nav 含 NaN → ValueError（不许静默 dropna），layer_table 同样要报错
  ⑦ layer_table：每层一行、指标齐全、与单指标函数一致；ls_summary 三种定义齐全
  ⑧ 断言条数守卫（防止测试空转）

用法（项目根目录）: python tests/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from backtest import metrics as MT
except ImportError as e:
    print(f"❌ 还没写 backtest/metrics.py（{e}）")
    print("   先照《metrics.py 教学参考》§7 写，再跑本测试")
    sys.exit(1)

passed = 0
failed = 0


def check(cond, label):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}")


def near(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


TD = MT.TRADING_DAYS   # 不做硬编码副本：年化常数只认 metrics.py 里那一处（A 股 244，见 PITFALLS #21）

print("── ① 常数日收益：年化 / 波动 / 夏普 ──")
r = 0.001
navA = pd.Series((1 + r) ** np.arange(TD + 1),
                 index=pd.date_range("2020-01-01", periods=TD + 1, freq="B"))
exp_ann = float(navA.iloc[-1] / navA.iloc[0]) ** (TD / len(navA)) - 1.0
got_ann = MT.ann_return(navA)
check(near(got_ann, exp_ann, 1e-12), f"几何年化 = {got_ann:.6f}（手算 {exp_ann:.6f}）")
check(MT.ann_vol(navA) < 1e-12, f"常数日收益 → 波动 ≈ 0（{MT.ann_vol(navA):.2e}）")
nav_flat = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=20, freq="B"))
check(np.isnan(MT.sharpe(nav_flat)), "净值恒定（波动精确 = 0）→ 夏普必须是 NaN（不许 inf）")
check(np.isnan(MT.sharpe(navA)), "常数日收益的近零波动（1.9e-15）→ 也必须是 NaN（不许返回 1e14 的垃圾夏普）")

print("── ② 最大回撤 ──")
navB = pd.Series([1.0, 1.1, 0.88, 0.9], index=pd.date_range("2020-01-01", periods=4, freq="B"))
check(near(MT.max_drawdown(navB), 0.88 / 1.1 - 1.0, 1e-12),
      f"先涨到 1.1 再跌到 0.88 → 回撤 {MT.max_drawdown(navB):.4%}（手算 −20%）")
check(MT.max_drawdown(navB) < 0, "回撤符号必须是负")
navC = pd.Series([1.0, 1.05, 1.1, 1.2], index=navB.index)
check(MT.max_drawdown(navC) == 0.0, "单调上涨 → 回撤 = 0")

print("── ③ 成本口径：累计 / 年化 ──")
tv = pd.Series([0.5, 0.25], index=pd.date_range("2020-01-01", periods=2, freq="B"))
fe = pd.Series([0.001, 0.002], index=tv.index)
check(near(MT.turnover_total(tv), 0.75), "累计换手 = 0.75")
check(near(MT.fee_total(fe), 0.003), "累计费用 = 0.003")
check(near(MT.turnover_ann(tv, navA), 0.75 / (len(navA) / TD), 1e-12), "年化换手 = 累计 ÷ 年数")

print("── ④ IC：同序 +1 / 反序 −1 / 无关 0.2、0.8 / 截面不足跳过 ──")
pidx = pd.DatetimeIndex(["2021-01-29", "2021-02-26", "2021-03-31"])
codes = ["A", "B", "C", "D"]
close = pd.DataFrame(10.0, index=pidx, columns=codes)
close.loc["2021-02-26"] = [20.0, 15.0, 12.0, 11.0]      # 收益 A > B > C > D
close.loc["2021-03-31"] = [30.0, 20.0, 14.0, 12.0]      # 区间收益仍是 A > B > C > D
ok = pd.DataFrame(True, index=pidx, columns=codes)
panels_ic = {"hfq_close": close, "can_trade": ok}
f_same = pd.DataFrame({"A": [4.0, 4.0], "B": [3.0, 3.0], "C": [2.0, 2.0], "D": [1.0, 1.0]}, index=pidx[:2])

ic_same = MT.rank_ic(f_same, panels_ic, pidx, min_names=3)
check(len(ic_same) == 2, f"IC 期数 = 调仓日数 − 1 = {len(ic_same)}")
check(bool((ic_same - 1.0).abs().max() < 1e-12), f"因子与未来收益同序 → IC = +1（{list(ic_same.round(4))}）")

f_rev = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0], "D": [4.0, 4.0]}, index=pidx[:2])
ic_rev = MT.rank_ic(f_rev, panels_ic, pidx, min_names=3)
check(bool((ic_rev + 1.0).abs().max() < 1e-12), f"反序 → IC = −1（{list(ic_rev.round(4))}）")

close2 = close.copy()
close2.loc["2021-02-26"] = [20.0, 11.0, 12.0, 15.0]      # 收益序变成 A > D > C > B
ic_none = MT.rank_ic(f_same, {"hfq_close": close2, "can_trade": ok}, pidx, min_names=3)
# 手算（Spearman ρ = 1 − 6Σd²/(n(n²−1))，n = 4）：第 1 期 Σd² = 8 → 0.2；第 2 期 Σd² = 2 → 0.8
check(near(float(ic_none.iloc[0]), 0.2, 1e-12) and near(float(ic_none.iloc[1]), 0.8, 1e-12),
      f"无关截面：手算 ρ = 0.2 / 0.8，实测 {list(ic_none.round(4))}")

ok_masked = ok.copy()
ok_masked.loc["2021-01-29", "D"] = False
ic_skip = MT.rank_ic(f_same, {"hfq_close": close, "can_trade": ok_masked}, pidx, min_names=4)
check(len(ic_skip) == 1, "截面不足 min_names → 该期跳过（不算、也不报错）")

print("── ⑤ 多空：标准复利三口径（期望值全部来自恒等式/手算，不用实现重算）──")
# 2026-11-03 T-3.4：删掉 ls_ratio / ls_diff / ls_month（比值法是净值比不是收益、
# 算术差漏波动拖累）。新的 ls_summary 出三个口径，下面每条都用【独立来源】验证：
#   · 多空累计 = 1.01^12 − 1        ← 复利定义，手算
#   · n_period == periods_per_year  ⇒ 年化必须【等于】累计（年化是恒等变换）
#   · 两层净值相同时，层差必须【恰好 0】   ← 恒等式，不依赖任何公式
#   · t 用 [0.02, 0.00, 0.01] ⇒ mean=.01, sd=.01 ⇒ t=√3   ← 手算
midx = pd.date_range("2020-01-31", periods=13, freq="ME")
# 12 期：层0 全程不动，层4 每期 +1%
nav_ok = pd.DataFrame({L: np.ones(13) for L in range(5)}, index=midx)
nav_ok[4] = 1.01 ** np.arange(13)
ls = MT.ls_summary(nav_ok, midx, borrow_cost=0.08, periods_per_year=12)
exp_cum = 1.01 ** 12 - 1.0
check(near(ls["多空累计"], exp_cum, 1e-12),
      f"多空累计 = 1.01^12−1（手算 {exp_cum:.8f}，实测 {ls['多空累计']:.8f}）")
check(near(ls["多空年化"], exp_cum, 1e-12),
      "12 期 / periods_per_year=12 ⇒ 年化必须等于累计（恒等式）")
check(near(ls["多空净年化"], exp_cum - 0.08, 1e-12),
      f"多空净 = 年化 − borrow_cost（{exp_cum - 0.08:.8f}）")
check(ls["期数"] == 12, f"期数 = {ls['期数']}（应 12）")
check(ls["层差(pp)"] > 0.0, f"层4 涨、层0 平 ⇒ 层差必须 > 0（实测 {ls['层差(pp)']:.4f}）")

# 恒等式：两层净值完全相同 ⇒ 层差恰好 0、多空累计恰好 0
nav_flat = pd.DataFrame({L: np.ones(13) for L in range(5)}, index=midx)
ls_flat = MT.ls_summary(nav_flat, midx, periods_per_year=12)
check(abs(ls_flat["层差(pp)"]) < 1e-12, "两层净值相同 ⇒ 层差必须恰好 0")
check(abs(ls_flat["多空累计"]) < 1e-12, "两层净值相同 ⇒ 多空累计必须恰好 0")

# t 值手算：diff = [+2%, 0%, +1%] ⇒ mean=.01, sd(ddof=1)=.01, t = .01/(.01/√3) = √3
tidx = pd.date_range("2020-01-31", periods=4, freq="ME")
nav_t = pd.DataFrame({L: np.ones(4) for L in range(5)}, index=tidx)
nav_t[4] = [1.0, 1.02, 1.02, 1.02 * 1.01]
ls_t = MT.ls_summary(nav_t, tidx, periods_per_year=12)
check(near(ls_t["多空t"], 3 ** 0.5, 1e-9),
      f"t = √3 = {3 ** 0.5:.6f}（手算），实测 {ls_t['多空t']:.6f}")
check(near(ls_t["多空p"], MT._norm_p(3 ** 0.5), 1e-12), "p = 正态近似(t=√3)")

# A 断言（PITFALLS #23 守卫）：rdates 里混进非交易日 ⇒ reindex 出 NaN ⇒ 必须拒绝
raised = False
try:
    MT._period_returns(nav_ok[4], [midx[0], pd.Timestamp("2020-02-15"), midx[1]])
except AssertionError:
    raised = True
check(raised, "_period_returns：rdates 含非交易日 ⇒ 断言拒绝（PITFALLS #23 守卫）")

print("── ⑥ nav 含 NaN → 报错，不许静默 dropna ──")
nav_bad = navA.copy()
nav_bad.iloc[10] = np.nan
raised = False
try:
    MT.ann_vol(nav_bad)
except ValueError:
    raised = True
check(raised, "ann_vol(nav 含 NaN) → ValueError")
raised = False
try:
    MT.layer_table(pd.DataFrame({0: nav_bad}))
except ValueError:
    raised = True
check(raised, "layer_table(nav 含 NaN) → ValueError")

print("── ⑦ layer_table / ls_summary ──")
# 层 0 = 空头（1.0 → 0.9）、层 4 = 多头（1.0 → 1.2）
_navS = pd.Series([1.0, 0.95, 0.9], index=pd.date_range("2020-01-01", periods=3, freq="B"))
_navL = pd.Series([1.0, 1.1, 1.2], index=_navS.index)
nav_df = pd.DataFrame({0: _navS, 4: _navL})
tab = MT.layer_table(nav_df)
check(list(tab.index) == [0, 4] and {"期末净值", "年化", "波动", "最大回撤", "夏普"} <= set(tab.columns),
      f"layer_table：每层一行、指标齐全（{list(tab.columns)}）")
check(near(float(tab.loc[0, "期末净值"]), 0.9, 1e-12) and near(float(tab.loc[4, "期末净值"]), 1.2, 1e-12),
      "layer_table 的期末净值与输入一致")
check(near(float(tab.loc[4, "最大回撤"]), MT.max_drawdown(_navL), 1e-12)
      and near(float(tab.loc[0, "年化"]), MT.ann_return(_navS), 1e-12),
      "layer_table 的值与单指标函数一致")

# ls_summary 的键集合变了（比值法/日差法/月差法 → 三口径 + t/p + 期数）
lsv = MT.ls_summary(nav_df, [_navL.index[0], _navL.index[-1]])
# 2026-11-11（T-3.7）：加 Newey-West 三件套（t / p / lags）——
#   多空收益序列有自相关，朴素 t 被系统性高估，所以两列必须都出。
check(set(lsv) == {'层差(pp)', '多空年化', '多空累计', '多空净年化', '多空t', '多空p',
                    '多空夏普', '多空最大回撤', '期数',
                    '多空t_nw', '多空p_nw', 'nw_lags'},
      'ls_summary 的键：%r' % (sorted(lsv),))
# lag 必须是非负整数（经验法则 4*(T/100)^(2/9)）
check(isinstance(lsv['nw_lags'], int) and lsv['nw_lags'] >= 0,
      'nw_lags 必须是非负整数（实测 %r）' % (lsv['nw_lags'],))
# 手算：只取首尾两点 ⇒ 多头 +20%、空头 −10% ⇒ 每期差 = 0.2 − (−0.1) = 0.3
check(near(lsv["多空累计"], 0.30, 1e-12),
      f"多空累计 = 0.2 − (−0.1) = 30%（实测 {lsv['多空累计']:.6%}）")
check(lsv["期数"] == 1, f"只取首尾两点 ⇒ 1 期（实测 {lsv['期数']}）")
check(lsv["层差(pp)"] > 0, "多头涨、空头跌 ⇒ 层差必须 > 0")
check("比值法" not in lsv, "比值法必须已删除（它是净值比不是收益）")

print("\n── equal_weight_nav 的 mask（§4.2）──")
# 基准必须跟宇宙走：用大盘组合去比小盘基准，超额会被系统性吞掉。
# 这里用手算的极小例子钉住“ mask 真的变了基准”。
_bt = pd.date_range("2020-01-01", periods=3, freq="B")
_r2 = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=_bt)
_pn2 = {"ret_1d_hfq": _r2, "can_trade": pd.DataFrame(True, index=_bt, columns=["A", "B"])}
_n_all = MT.equal_weight_nav(_pn2)
_maskB = pd.DataFrame({"A": [False] * 3, "B": [True] * 3}, index=_bt)
_n_bsel = MT.equal_weight_nav(_pn2, mask=_maskB)
check(abs(float(_n_all.iloc[-1]) - 1.05) < 1e-12,
      f"无 mask ⇒ 含 A 等权 ⇒ {float(_n_all.iloc[-1]):.4f}（手算 1.05）")
check(abs(float(_n_bsel.iloc[-1]) - 1.0) < 1e-12,
      f"mask 剔掉 A ⇒ 只剩 B(0) ⇒ {float(_n_bsel.iloc[-1]):.4f}（手算 1.0）")
check(abs(float(_n_all.iloc[-1]) - float(_n_bsel.iloc[-1])) > 1e-6, "mask 必须真的改变基准")

print("\n" + "=" * 46)
print('\n── ⑧ 被白吃的 alpha（T-2.4）：被挡的单子错过了多少 ──')
# 手工造的假 res —— 不跑 engine，纯测数学 ✓
_EX = pd.DatetimeIndex(["2020-01-03", "2020-01-06"])


def _fake_res(buy, sell, nav=1.0):
    return {"blocked_buy": pd.DataFrame(buy, index=_EX, columns=["A"]),
            "blocked_sell": pd.DataFrame(sell, index=_EX, columns=["A"]),
            "nav_pre": pd.Series(nav, index=_EX)}


_FLAT = {"hfq_close": pd.DataFrame([[10.0], [11.0]], index=_EX, columns=["A"]),
         "hfq_open": pd.DataFrame([[10.0], [11.0]], index=_EX, columns=["A"])}

pp = MT._blocked_pp(_fake_res([[0.05], [0.0]], [[0.0], [0.0]]), _FLAT, periods_per_year=12)
exp = 0.05 * 10 * 0.10 / 1.0 / 2 * 12 * 100     # 每期损失率 → 平均 → 年化 → %
check(near(pp, exp, 1e-9), f"涨停挡掉 + 下期涨 10% ⇒ {pp:.6f}（手算 {exp:.6f}）")
check(pp > 0.0, "少赚了 ⇒ 必须是正数")

pp0 = MT._blocked_pp(_fake_res([[0.0], [0.0]], [[0.0], [0.0]]), _FLAT, periods_per_year=12)
check(abs(pp0) < 1e-12, f"什么都没挡 ⇒ 必须恰好 0（实际 {pp0:.3e}）")

print("── ⑨ 报告新增字段（2026-11-04）：IC胜率 / 多空夏普 / 多空回撤 / 每期换手 ──")

# ── IC>0 胜率：手算 2/4 ──
_ic = pd.Series([0.10, -0.05, 0.20, 0.00])
_s = MT.ic_summary(_ic)
check(near(_s["ic_pos_rate"], 0.5, 1e-12),
      f"IC>0 胜率 = 2/4 = 0.5（实测 {_s.get('ic_pos_rate')}；0.00 不算正数）")
check(near(_s["ic_mean"], 0.0625, 1e-12), "ic_mean 同时复核（(0.1-0.05+0.2+0)/4 = 0.0625）")
check(_s["n_pos"] == 2 and _s["n"] == 4, f"n_pos 直接返回 = 2（实测 {_s.get('n_pos')}）")

# IC-IR 的年化粒度必须跟着 periods_per_year：√4 / √12 差 = 2 倍
_s4 = MT.ic_summary(_ic, periods_per_year=4)
_s12 = MT.ic_summary(_ic, periods_per_year=12)
check(near(_s12["ic_ir"] / _s4["ic_ir"], (12 / 4) ** 0.5, 1e-12),
      f"IC-IR(ppy=12) / IC-IR(ppy=4) 必须 = √3 = {(12/4)**0.5:.4f}"
      f"（实测 {_s12['ic_ir']/_s4['ic_ir']:.4f}）")

# 恒定 IC ⇒ sd≈0 ⇒ IC-IR 必须 NaN（不许印 1e16 的垃圾）
_sf = MT.ic_summary(pd.Series([0.05] * 10), periods_per_year=12)
check(np.isnan(_sf["ic_ir"]), f"IC 恒定 ⇒ IC-IR 必须 NaN（实测 {_sf['ic_ir']}）")

# ── 多空最大回撤：diff = [+10%, −20%, +10%] ──
#    净值 1.0 → 1.1 → 0.88 → 0.968   峰 1.1、谷 0.88 ⇒ 回撤 = 0.88/1.1 − 1 = −20%（精确）
_midx = pd.date_range("2020-01-31", periods=4, freq="ME")
_nv = pd.DataFrame({0: np.ones(4), 4: np.array([1.0, 1.10, 0.88, 0.968])}, index=_midx)
_ls = MT.ls_summary(_nv, list(_midx), periods_per_year=12)
check(near(_ls["多空最大回撤"], -0.20, 1e-12),
      f"多空最大回撤 = 0.88/1.1 − 1 = −20%（实测 {_ls['多空最大回撤']:.6f}）")
check(-1.0 <= _ls["多空最大回撤"] <= 0.0, "回撤必须落在 [−1, 0]")

# ── 多空夏普：每期收益恒定 ⇒ 标准差 0 ⇒ 必须是 NaN（不许 inf）──
_nv2 = pd.DataFrame({0: np.ones(13), 4: 1.01 ** np.arange(13)},
                    index=pd.date_range("2020-01-31", periods=13, freq="ME"))
_ls2 = MT.ls_summary(_nv2, list(_nv2.index), periods_per_year=12)
check(np.isnan(_ls2["多空夏普"]), f"每期恒定收益 ⇒ 多空夏普必须 NaN（实测 {_ls2['多空夏普']}）")

# ── 等均值但带波动 ⇒ 复利拖累 ⇒ 多空年化和夏普都必须是负的 ──
_r = np.array([0.02, -0.02] * 6)                      # 12 期，算术均值 0
_nv3 = pd.DataFrame({0: np.ones(13), 4: np.concatenate([[1.0], np.cumprod(1 + _r)])},
                    index=pd.date_range("2020-01-31", periods=13, freq="ME"))
_ls3 = MT.ls_summary(_nv3, list(_nv3.index), periods_per_year=12)
check(_ls3["多空年化"] < 0, f"算术均值 0 + 波动 ⇒ 复利为负（实测 {_ls3['多空年化']:.4%}）")
check(_ls3["多空夏普"] < 0, "同样的道理 ⇒ 夏普必须为负")

# ── 每期换手：累计 6.0 / 2 期 = 3.0 ──
_nv4 = pd.DataFrame({0: np.ones(3), 4: np.ones(3)},
                    index=pd.date_range("2020-01-31", periods=3, freq="ME"))
_tab = MT.layer_table(_nv4, {0: {"turnover": pd.Series([2.0]), "fees": pd.Series([0.0])},
                             4: {"turnover": pd.Series([2.0]), "fees": pd.Series([0.0])}})
check("每期换手" in _tab.columns, "layer_table 多出「每期换手」列")
check(near(float(_tab.loc[0, "每期换手"]), 2.0, 1e-12),
      f"1 期累计 2.0 ⇒ 每期 2.0（实测 {_tab.loc[0, '每期换手']}）")

check(passed + failed >= 20, f"断言条数守卫：{passed + failed} 条")
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print("有断言未通过 ❌")
    sys.exit(1)
print("全部通过 ✅ 收益 / 风险 / 成本 / IC / 三种多空 可用（下一步：report）")
