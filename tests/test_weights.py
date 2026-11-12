# -*- coding: utf-8 -*-
"""
tests/test_weights.py —— weights.py 验收（手工 fixture：10 只 × 3 个调仓日）

  ① W[L].index == layers.index == rdates（只能有调仓日）
  ② dtype：W = float32、layers = int8
  ③ 方向：因子最大的票在**层 4**、最小的在**层 0**
  ④ 每层只数差 ≤ 1（10 只 → 2/2/2/2/2；9 只 → 2/2/2/2/1）
  ⑤ 每个调仓日每层权重和 = 1（该层当天为空 → 0）
  ⑥ W[L] 非零 ⟺ layers == L
  ⑦ 不可交易的票：层号 -1、权重 0
  ⑧ 因子全 NaN 那天：全部 -1 / 0，且**不崩**
  ⑨ 不改动入参（factor / can_trade 快照对比）
  ➕ 并列值：两只同值 → 仍被拆到相邻层，层大小不破
  ➕ 层内等权手算：层 4 有 2 只 → 各 0.5

用法（项目根目录）: python tests/test_weights.py
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
    from backtest import weights as WT
except ImportError as e:
    print(f"❌ 还没写 backtest/weights.py（{e}）")
    print("   先照《weights.py 教学参考》§7 写，再跑本测试")
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


RDATES = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
CODES = list("ABCDEFGHIJ")


def make_inputs():
    """d1：因子 1..10 递增（A 最小、J 最大），全部可交易
       d2：因子 10..1 递减（A 最大），**A 不可交易** → 只剩 9 只
       d3：因子全 NaN → 一只都分不了层"""
    factor = pd.DataFrame(
        [np.arange(1, 11, dtype="float64"),
         np.arange(10, 0, -1, dtype="float64"),
         np.full(10, np.nan)],
        index=RDATES, columns=CODES)
    can = pd.DataFrame(True, index=RDATES, columns=CODES)
    can.loc[RDATES[1], "A"] = False
    return factor, can


F0, CAN0 = make_inputs()
W, LAY = WT.build_target_weights(F0, list(RDATES), CAN0, n_layers=5)
N = len(W)
print(f"fixture：{len(RDATES)} 个调仓日 × {len(CODES)} 只 ｜ 分 {N} 层")

print("\n── ① index 只能是调仓日 ──")
check(all(list(w.index) == list(RDATES) for w in W) and list(LAY.index) == list(RDATES),
      f"W[0..{N-1}] 与 layers 的 index 都是那 {len(RDATES)} 个调仓日")

print("\n── ② dtype ──")
check(all(str(w.dtypes.iloc[0]) == "float32" for w in W) and str(LAY.dtypes.iloc[0]) == "int8",
      f"W = {W[0].dtypes.iloc[0]} ｜ layers = {LAY.dtypes.iloc[0]}")

print("\n── ③ 方向：因子最大 → 层 {0}、最小 → 层 0 ──".format(N - 1))
check(int(LAY.loc[RDATES[0], "J"]) == N - 1 and int(LAY.loc[RDATES[0], "A"]) == 0,
      f"d1：J(值10)={int(LAY.loc[RDATES[0], 'J'])}（应 {N-1}）｜ A(值1)={int(LAY.loc[RDATES[0], 'A'])}（应 0）")
check(int(LAY.loc[RDATES[1], "B"]) == N - 1 and int(LAY.loc[RDATES[1], "A"]) == -1,
      f"d2：A 不可交易 → 层 -1={int(LAY.loc[RDATES[1], 'A'])}；剩下最大的 B(值9) 进层 {int(LAY.loc[RDATES[1], 'B'])}")

print("\n── ④ 每层只数差 ≤ 1 ──")
for dt, want in ((RDATES[0], [2, 2, 2, 2, 2]), (RDATES[1], [2, 2, 2, 2, 1])):
    sizes = [int((LAY.loc[dt] == L).sum()) for L in range(N)]
    check(max(sizes) - min(sizes) <= 1 and sum(sizes) == sum(want),
          f"{str(dt.date())}：层大小 {sizes}（合计 {sum(sizes)} 只可交易）")

print("\n── ⑤ 每层权重和 = 1（空层 = 0）──")
bad = []
for L in range(N):
    for dt in RDATES:
        s = float(W[L].loc[dt].sum())
        members = int((LAY.loc[dt] == L).sum())
        if members == 0:
            if s != 0.0:
                bad.append((L, str(dt.date()), s))
        elif abs(s - 1.0) > 1e-6:
            bad.append((L, str(dt.date()), s))
check(not bad, f"每层每个调仓日：非空 → 1.0000，空 → 0（异常 {bad[:3]}）")

print("\n── ⑥ W[L] 非零 ⟺ layers == L ──")
check(all(((w != 0) == (LAY == L)).to_numpy().all() for L, w in enumerate(W)),
      "权重矩阵与层号完全一致（不会出现'有权重但层号不对'）")

print("\n── ⑦ 不可交易 → 层 -1 且权重 0 ──")
check(int(LAY.loc[RDATES[1], "A"]) == -1 and all(float(w.loc[RDATES[1], "A"]) == 0.0 for w in W),
      "d2 的 A：层 -1、五层权重全 0")

print("\n── ⑧ 因子全 NaN 那天不崩、全部 -1 / 0 ──")
check(bool((LAY.loc[RDATES[2]] == -1).all()) and all(float(w.loc[RDATES[2]].sum()) == 0.0 for w in W),
      f"d3：层号全 -1，权重全 0（没崩）")

print("\n── ⑨ 不改动入参 ──")
check(F0.equals(make_inputs()[0]) and CAN0.equals(make_inputs()[1]), "factor / can_trade 一格未改")

print("\n── ➕ 加测 ──")
f2, can2 = make_inputs()
f2.loc[RDATES[0], ["B", "C"]] = 5.5                       # 制造并列
W2, L2 = WT.build_target_weights(f2, list(RDATES), can2, n_layers=5)
sizes2 = [int((L2.loc[RDATES[0]] == L).sum()) for L in range(N)]
check(int(L2.loc[RDATES[0], "B"]) != int(L2.loc[RDATES[0], "C"]) and max(sizes2) - min(sizes2) <= 1,
      f"并列值 B=C=5.5 → 被拆到相邻层（{int(L2.loc[RDATES[0], 'B'])} / {int(L2.loc[RDATES[0], 'C'])}），层大小 {sizes2}")
check(abs(float(W[N - 1].loc[RDATES[0], "I"]) - 0.5) < 1e-6 and abs(float(W[N - 1].loc[RDATES[0], "J"]) - 0.5) < 1e-6,
      f"层 {N-1} 只有 I / J 两只 → 各 {float(W[N-1].loc[RDATES[0], 'I']):.4f}")

print("\n── ⑦ select（选股池）—— 2026-11-12 ──")
# ① select=None 必须与不传【逐位相同】（回归门：默认行为一个字都不许变）
W_a, L_a = WT.build_target_weights(F0, list(RDATES), CAN0, n_layers=5)
W_b, L_b = WT.build_target_weights(F0, list(RDATES), CAN0, n_layers=5, select=None)
_same = all(np.array_equal(W_a[L].to_numpy(), W_b[L].to_numpy()) for L in range(len(W_a)))
check(_same and np.array_equal(L_a.to_numpy(), L_b.to_numpy()),
      "select=None 与不传 select 的权重/分层逐位相同")

# ② 收窄 select：只留 A~E 五只 ⇒ 未选中的票层 = -1，每层权重和仍为 1
sel = pd.DataFrame(False, index=RDATES, columns=CODES)
sel.loc[:, ["A", "B", "C", "D", "E"]] = True
W_s, L_s = WT.build_target_weights(F0, list(RDATES), CAN0, n_layers=5, select=sel)
_outs = [c for c in CODES if c not in ("A", "B", "C", "D", "E")]
check(all(int(L_s.loc[RDATES[0], c]) == -1 for c in _outs),
      f"select 外的 {len(_outs)} 只（F~J）层 = -1")
check(all(abs(float(w.loc[RDATES[0]].sum()) - 1.0) < 1e-6 for w in W_s),
      "收窄后每层权重和仍为 1（选股池变小不改变归一化）")
check(int((L_s.loc[RDATES[0]] == -1).sum()) == len(_outs),
      f"层 -1 的只数 = {int((L_s.loc[RDATES[0]] == -1).sum())}（应 {len(_outs)}）")

# ③ 健康断言只看 can_trade，不被 select 影响
#    造 1201 列（>1000 才会跑比例断言）：can_trade 占 50%（健康），
#    select 只留 8% ⇒ 若错用了 select 做断言就会误报 "ratio 0.08 out of range"
_nn = 1201
_big_codes = [f"S{i:04d}" for i in range(_nn)]
_fac = pd.DataFrame(np.tile(np.arange(_nn, dtype="float64"), (2, 1)),
                    index=RDATES[:2], columns=_big_codes)
_can = pd.DataFrame(True, index=RDATES[:2], columns=_big_codes)
_can.iloc[:, _nn // 2:] = False                      # 50% 可交易（健康区间内）
_sel = pd.DataFrame(False, index=RDATES[:2], columns=_big_codes)
_sel.iloc[:, :100] = True                            # 只留 100/1201 ≈ 8%
try:
    WT.build_target_weights(_fac, list(RDATES[:2]), _can, n_layers=5, select=_sel)
    check(True, "can_trade 占 50%（健康）+ select 只留 8% ⇒ 不误报（断言只看 can_trade）")
except AssertionError as e:
    check(False, f"select 把健康断断言带偏了：{str(e)[:80]}")

# ④ 哨兵没被削弱：can_trade 自己退化（20%）⇒ 仍必须拒跑
_can_bad = pd.DataFrame(True, index=RDATES[:2], columns=_big_codes)
_can_bad.iloc[:, _nn // 5:] = False                  # 20% < 0.4
try:
    WT.build_target_weights(_fac, list(RDATES[:2]), _can_bad, n_layers=5, select=_sel)
    check(False, "can_trade 占 20% 竟然没报错 —— 哨兵被削弱了")
except AssertionError:
    check(True, "can_trade 占 20% ⇒ 仍拒跑（哨兵没被削弱）")


print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print(f"仍有 {failed} 项失败 ❌ 按上面提示修 weights.py")
    sys.exit(1)
print("全部通过 ✅ 分层目标权重可用（下一步：engine 撮合）")
