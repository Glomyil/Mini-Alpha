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

print("── ⑤ 三种多空定义：各自手算，且互不相等 ──")
navL = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2020-01-01", periods=3, freq="B"))
navS = pd.Series([1.0, 0.95, 0.9], index=navL.index)
ratio = MT.ls_ratio(navL, navS)
diff = MT.ls_diff(navL, navS)
month = MT.ls_month(navL, navS, [navL.index[0], navL.index[-1]])
exp_diff = (0.1 + 0.05) + (1.2 / 1.1 - 0.9 / 0.95)
check(near(ratio, 1.2 / 0.9 - 1.0, 1e-12), f"比值法 = {ratio:.4%}（手算 {1.2 / 0.9 - 1:.4%}）")
check(near(diff, exp_diff, 1e-12), f"日差法 = {diff:.4%}（手算 {exp_diff:.4%}）")
check(near(month, 0.3, 1e-12), f"月差法 = {month:.4%}（手算 30%（只在两端取区间收益））")
check(abs(ratio - diff) > 1e-6 and abs(ratio - month) > 1e-6 and abs(diff - month) > 1e-6,
      "三种定义两两不相等（防止把三种写成一个）")

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
nav_df = pd.DataFrame({0: navS, 4: navL})          # 层 0 = 空头、层 4 = 多头
tab = MT.layer_table(nav_df)
check(list(tab.index) == [0, 4] and {"期末净值", "年化", "波动", "最大回撤", "夏普"} <= set(tab.columns),
      f"layer_table：每层一行、指标齐全（{list(tab.columns)}）")
check(near(float(tab.loc[0, "期末净值"]), 0.9, 1e-12) and near(float(tab.loc[4, "期末净值"]), 1.2, 1e-12),
      "layer_table 的期末净值与输入一致")
check(near(float(tab.loc[4, "最大回撤"]), MT.max_drawdown(navL), 1e-12)
      and near(float(tab.loc[0, "年化"]), MT.ann_return(navS), 1e-12),
      "layer_table 的值与单指标函数一致")
lsv = MT.ls_summary(nav_df, [navL.index[0], navL.index[-1]])
check(set(lsv) == {"比值法", "日差法", "月差法"} and near(lsv["比值法"], 1.2 / 0.9 - 1.0, 1e-12)
      and near(lsv["月差法"], 0.3, 1e-12),
      "ls_summary：三种定义齐全，默认最后一列=多头 / 第一列=空头")

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

pp = MT._blocked_pp(_fake_res([[0.05], [0.0]], [[0.0], [0.0]]), _FLAT)
exp = 0.05 * 10 * 0.10 / 1.0 / 2 * 12 * 100     # 每期损失率 → 平均 → 年化 → %
check(near(pp, exp, 1e-9), f"涨停挡掉 + 下期涨 10% ⇒ {pp:.6f}（手算 {exp:.6f}）")
check(pp > 0.0, "少赚了 ⇒ 必须是正数")

pp0 = MT._blocked_pp(_fake_res([[0.0], [0.0]], [[0.0], [0.0]]), _FLAT)
check(abs(pp0) < 1e-12, f"什么都没挡 ⇒ 必须恰好 0（实际 {pp0:.3e}）")

check(passed + failed >= 20, f"断言条数守卫：{passed + failed} 条")
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print("有断言未通过 ❌")
    sys.exit(1)
print("全部通过 ✅ 收益 / 风险 / 成本 / IC / 三种多空 可用（下一步：report）")
