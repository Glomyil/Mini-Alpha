# -*- coding: utf-8 -*-
"""ops/ab_valuation.py —— 口径 A/B：收盘估值 vs 开盘估值（只读，不落盘）

为什么需要它
    engine 的主口径 `nav` 是**收盘估值**（每天收盘后的账户值）；
    诊断列 `nav_open` 是**开盘估值**（每天开盘时刻的账户值，与调仓同一时点）。
    两者之差 = **口径敏感度**：差得小 → 结论稳健；差得大 → 报告里必须写清用的是哪条口径。

顺带诊断（这里能直接量出来）
    「执行日 持仓市值 / 净值」= 杠杆。> 1 说明有票卖不掉、钱却被拿去买了别的票（TECH_DEBT DEBT-001）。

用法：python ops/ab_valuation.py [因子名]      # 默认 rev_20d
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from backtest.panel import load_panels
from backtest.assemble import build_features, rebalance_dates
from backtest.factors import FACTORS
from backtest.weights import weights_for_factor
from backtest.engine import run_all_layers, nav_frame

FAC = sys.argv[1] if len(sys.argv) > 1 else "rev_20d"
TD = 244          # 一年多少个交易日：A 股口径（242~244）；252 是美股口径 —— 见 PITFALLS #21


def ann_return(nav):
    return float(nav.iloc[-1] / nav.iloc[0]) ** (TD / len(nav)) - 1.0


def ann_vol(nav):
    return float(nav.pct_change(fill_method=None).dropna().std(ddof=1) * np.sqrt(TD))


def max_dd(nav):
    return float((nav / nav.cummax() - 1.0).min())


def sharpe(nav):
    v = ann_vol(nav)
    return float("nan") if v == 0 else float(ann_return(nav) / v)


def ls_ratio(a, b):
    return float(a.iloc[-1] / b.iloc[-1]) - 1.0


def ls_diff(a, b):
    return float((a.pct_change(fill_method=None) - b.pct_change(fill_method=None)).dropna().sum())


def ls_month(a, b, rd):
    ra = a.reindex(rd).pct_change(fill_method=None)
    rb = b.reindex(rd).pct_change(fill_method=None)
    return float((ra - rb).dropna().sum())


def table(nav, label):
    rows = []
    for k in nav.columns:
        n = nav[k].dropna()
        rows.append({"层": k, "期末净值": n.iloc[-1], "年化": ann_return(n),
                     "波动": ann_vol(n), "最大回撤": max_dd(n), "夏普": sharpe(n)})
    tab = pd.DataFrame(rows).set_index("层")
    print(f"── {label} ──")
    print(tab.round(4).to_string())
    return tab


panels = load_panels()
rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= 2016]
W, layers = weights_for_factor(FAC, panels, rdates)
res = run_all_layers(W, panels, rdates)

nav_c = nav_frame(res)                                              # 收盘口径（主）
nav_o = pd.DataFrame({k: r["nav_open"] for k, r in enumerate(res)})  # 开盘口径（诊断）

print(f"因子 {FAC} ｜ 调仓日 {len(rdates)} ｜ 交易日 {len(nav_c)}")
print()
t_c = table(nav_c, "收盘口径（主口径）")
print()
t_o = table(nav_o, "开盘口径（诊断）")
print()
print("── 多空（三种定义）──")
for label, nav in (("收盘", nav_c), ("开盘", nav_o)):
    a, b = nav[nav.columns[-1]], nav[nav.columns[0]]
    print(f"{label}: 比值法 {ls_ratio(a, b):.2%} ｜ 日差法 {ls_diff(a, b):.2%} ｜ 月差法 {ls_month(a, b, rdates):.2%}")
print()
print("── 口径敏感度（开盘 − 收盘）──")
print(pd.DataFrame({"年化差": t_o["年化"] - t_c["年化"],
                    "回撤差": t_o["最大回撤"] - t_c["最大回撤"],
                    "期末净值比": t_o["期末净值"] / t_c["期末净值"]}).round(4).to_string())
print()
print("── 超配 / 费用诊断（每层）──")
for k, r in enumerate(res):
    nav, cash = r["nav"], r["cash"]
    hold = nav - cash                                   # 恒等式：nav = 现金 + 持仓市值
    cumfee = r["fees"].cumsum().reindex(nav.index).ffill().fillna(0.0)
    borrowed = (-cash - cumfee).clip(lower=0)           # 真"借来的钱"（必须扣掉累计费用，否则测的是费用拖累）
    print(f"层 {k}: 持仓/净值 均值 {(hold / nav).mean():.4f} ｜ 真超配 均值 {borrowed.mean():.5f} "
          f"最大 {borrowed.max():.5f} ｜ 累计费用 {r['fees'].sum():.5f} "
          f"｜ 超配>0 天数 {int((borrowed > 1e-9).sum())}/{len(nav)}")
print()
print("读法：年化差 < 0.5pp → 结论稳健；多空差大 → 口径必须写进报告；杠杆 >1 说明有票卖不掉却被当成买入资金（DEBT-001）")
