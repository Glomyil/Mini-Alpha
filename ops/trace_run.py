# -*- coding: utf-8 -*-
"""ops/trace_run.py —— 端到端追踪：一只股票、一个调仓日，走完 长表 → 面板 → 指标 → 因子 → 权重 → 引擎输入

用法（项目根目录）：
    python ops/trace_run.py                                # 默认 sh.600579 / 2021-01-29 / rev_20d
    python ops/trace_run.py sz.000007 2021-01-29
    python ops/trace_run.py sz.000007 2021-01-29 rev_5d

只读：不改数据、不落盘（面板走缓存，秒级）。每一段都同时打印**手算值**与**代码里的值**，
对不上直接 ❌ —— 这就是"强制打通前面每一格"的那根线。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from backtest.panel import SRC, load_panels
from backtest.features import FEATURES
from backtest.factors import FACTORS
from backtest.assemble import build_features, rebalance_dates
from backtest.weights import weights_for_factor

CODE = sys.argv[1] if len(sys.argv) > 1 else "sh.600579"
DT = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2021-01-29")
FAC = sys.argv[3] if len(sys.argv) > 3 else "rev_20d"

ok_all = True


def show(label, manual, code_val, tol=1e-6):
    global ok_all
    same = (pd.isna(manual) and pd.isna(code_val)) or abs(float(manual) - float(code_val)) <= tol
    ok_all &= bool(same)
    mark = "✅" if same else "❌"
    print(f"  {mark} {label:<34} 手算 {manual!r:>18}   代码 {code_val!r:>18}")


print(f"标的 {CODE} ｜ 调仓日 {DT.date()} ｜ 因子 {FAC}")
print()

# ── ① 净数据长表：唯一产出者（close × back_adj_factor = hfq_close 等等）─────
d = pd.read_parquet(SRC / f"{CODE}.parquet")
d["date"] = pd.to_datetime(d["date"])
w = d[(d["date"] >= DT - pd.Timedelta(days=40)) & (d["date"] <= DT + pd.Timedelta(days=5))]
row = d[d["date"] == DT].iloc[0]
print("① 净数据长表（data/storage/cleaner_daily/%s.parquet，%d 列）" % (CODE, len(d.columns)))
print(w[["date", "close", "back_adj_factor", "hfq_close", "hfq_open", "ret_1d_hfq",
         "is_suspended", "is_price_bad"]].tail(4).to_string(index=False))
show("hfq_close = close × back_adj_factor", row["close"] * row["back_adj_factor"], row["hfq_close"], 1e-4)
print()

# ── ② 面板：4 张宽表里这条股票这一天 ────────────────────────────────────────
P = load_panels()
days = P["hfq_close"].index
i = int(days.get_loc(DT))
n = 20
print("② 面板（宽表 2839 × 5471，从缓存读回）")
print(f"   hfq_close[{DT.date()}] = {P['hfq_close'].iloc[i][CODE]:.4f} ｜ "
      f"hfq_close[-{n} 天] = {P['hfq_close'].iloc[i - n][CODE]:.4f}")
print(f"   hfq_open / ret_1d_hfq / can_trade = "
      f"{P['hfq_open'].iloc[i][CODE]:.4f} / {P['ret_1d_hfq'].iloc[i][CODE]:.6f} / {bool(P['can_trade'].iloc[i][CODE])}")
print()

# ── ③ 指标：features.ret(n)（算法唯一的家） ────────────────────────────────
spec = FACTORS[FAC]
feat_name, params = spec["needs"][0]
feats = build_features(P, [FAC], {})
tbl = feats[feat_name + "_" + "_".join(str(v) for _, v in sorted(params.items()))]
print(f"③ 指标表 {tbl.shape}（assemble 按 needs 现算 + 会话缓存）")
show(f"{feat_name}(n={params['n']}) 手算", P["hfq_close"].iloc[i][CODE] / P["hfq_close"].iloc[i - params["n"]][CODE] - 1,
     tbl.iloc[i][CODE], 1e-6)
print()

# ── ④ 因子：fn × sign（方向只在这里乘一次） ────────────────────────────────
val = spec["fn"](feats, {}) * spec["sign"]
print(f"④ 因子 {FAC}（fn = 取 {feat_name}_{params['n']}，sign = {spec['sign']}）")
show("因子值 手算", tbl.iloc[i][CODE] * spec["sign"], val.iloc[i][CODE], 1e-6)
print()

# ── ⑤ 权重：名次 → 层号 → 层内等权 ─────────────────────────────────────────
rd = [x for x in rebalance_dates(days, rule="ME") if x.year >= 2016]
W, L = weights_for_factor(FAC, P, rd)
v = val.loc[DT]
okmask = P["can_trade"].loc[DT].eq(True) & v.notna()
tk = v.index[okmask]
r = v[tk].rank(method="first")
layer_manual = int(((r[CODE] - 1) * 5) // len(tk))
print(f"⑤ 权重（{DT.date()}：可用 {len(tk)} 只 → 每层约 {len(tk) // 5} 只）")
print(f"   名次 {int(r[CODE])} / {len(tk)}  → 层号 手算 {layer_manual} vs layers 表 {int(L.loc[DT, CODE])}")
show("层内权重 手算", 1.0 / int((L.loc[DT] == layer_manual).sum()), W[layer_manual].loc[DT, CODE], 1e-9)
j = i + 1
print(f"   引擎输入：执行日 = {days[j].date()} ｜ hfq_open = {P['hfq_open'].iloc[j][CODE]} ｜ "
      f"can_trade = {bool(P['can_trade'].iloc[j][CODE])}")
w_t = float(W[layer_manual].loc[DT, CODE])
nav_pre = 1.0
px = float(P["hfq_open"].iloc[j][CODE])
show("目标金额 = nav_pre × w", nav_pre * w_t, w_t, 1e-12)
show("目标股数 = 目标金额 / 开盘价", nav_pre * w_t / px, nav_pre * w_t / px, 1e-12)
show("买入费用 = 金额 × 0.00025", nav_pre * w_t * 0.00025, nav_pre * w_t * 0.00025, 1e-15)
print()

# ── ⑥ 小结 ────────────────────────────────────────────────────────────────
print("=" * 78)
print(("✅ 全链条一致：长表 → 面板 → 指标 → 因子 → 权重 → 引擎输入 每一格都能手算复现"
       if ok_all else "❌ 有格子对不上 —— 顺着上面第一个 ❌ 往上查唯一产出者"))
print("下一步（engine）：用 ⑤ 的目标权重 + 执行日的 hfq_open + can_trade → 成交 / 现金 / 净值")
