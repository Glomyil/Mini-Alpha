# -*- coding: utf-8 -*-
"""
tests/test_engine.py —— engine.py 验收（手工 fixture：4 只 × 8 个交易日，全部数字手算）

  ① nav.index = 全部交易日；首日 nav = 1.0（成交发生在 t+1）
  ② trades.index = 两个执行日（非调仓日不动）
  ③ 执行日 1：买不到的票（can_trade=False）不成交 → 股数 0，钱留在现金
  ④ 执行日 1：换手 = 1/3（一侧口径 0.5×(买+卖)/成交前净值，手算）
  ⑤ 费用从现金扣、只扣一次：现金 = 1 − 2/3 − (2/3)×fee_buy；净值 = 1 − (2/3)×fee_buy
  ⑥ 非调仓日股数不变 → 价格不变则净值不变
  ⑦ 执行日 2：成交**股数变动** = 手算（目标股数 − 现有股数）→ 抓"用 close(10) 而不是 open(12/9) 当成交价"（错用 close，A 会从卖出变成买入）
  ⑧ 执行日 2：净值 = 手算（现金恒等式 + 费用 + 买卖方向 A 卖 B/C 买）
  ⑨ 停牌日估值必须 ffill（B 收盘价 NaN，净值不许归零）
  ⑩ 账本恒等式：nav = cash + Σ 股数 × 估值价
  ⑪ 未来函数：把 d6/d7 的价格 ×5 → d0–d5 的净值逐格不变
  ⑫ 入参（W / 面板）逐格未被改动
  ⑬ run_all_layers / nav_frame：形状 (8, 5)；全 0 权重层 → 净值恒 1、零成交
  ⑭ 全程无价的票（退市/无数据）：从不成交、不崩
  ⑮ 滑点：买价更贵 / 卖价更便宜（默认单边 5 bp），差额与成交额同量级
  ⑯ 成交前估值兜底不许取【当天收盘】（未来函数）：独立小 fixture —— R 当天开盘缺失但当天收盘 20，
     断言 S 的目标股数按"上一交易日收盘"算（不是按当天收盘算）
  ⑰ 末位边界：最后一个调仓日没有下一交易日 → 被跳过（真实数据 129 → 128）
  ⑱ 调仓日不在面板索引里（get_indexer = −1）→ 跳过不崩
  ⑲ 中途退市：按最后有效价永久计值（**钉住已知偏差**）
  ⑳ W 的列与面板列不一致：顺序打乱 / 子集 / 多余列（KeyError）
  ㉑ 现金透支：持仓卖不掉时的无成本融资（**钉住已知偏差**，杠杆 ≈ 2）
  ㉒ init_cash 尺度不变性（钉住「目前无整手 / 无最低佣金」）
  ㉓ 诊断列：nav_open（开盘口径逐日）+ nav_pre（成交前净值）→ 恒等式 nav_open(执行日) = nav_pre − 费用
  ㉔ 断言条数守卫（防止测试空转）

  注：基础 fixture 一律 slippage=0.0（手算干净）；滑点与未来函数各自单独一节验证。

用法（项目根目录）: python tests/test_engine.py
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
    from backtest import engine as EN
except ImportError as e:
    print(f"❌ 还没写 backtest/engine.py（{e}）")
    print("   先照《engine.py 教学参考》§7 写，再跑本测试")
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


DAYS = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
                         "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"])
CODES = ["A", "B", "C", "D"]
RDATES = [DAYS[0], DAYS[4]]                 # 调仓日 → 执行日 DAYS[1] 与 DAYS[5]
FB, FS = 0.00025, 0.00075                   # 买入 / 卖出费率（与参考实现默认值一致）


def make_inputs(scale_tail=1.0):
    """A/B/C 价格恒为 10；执行日 2 的 **开盘价** A=12、B=9；B 在 d6 停牌（收盘价 NaN）；
       D 全程无价；执行日 1 的 C 不可交易；d6/d7 的价格可整体缩放（测未来函数用）。"""
    close = pd.DataFrame(10.0, index=DAYS, columns=CODES)
    open_ = pd.DataFrame(10.0, index=DAYS, columns=CODES)
    open_.loc[DAYS[5], "A"] = 12.0
    open_.loc[DAYS[5], "B"] = 9.0
    open_.loc[DAYS[6]:, :] *= scale_tail
    close.loc[DAYS[6]:, :] *= scale_tail
    close.loc[DAYS[6], "B"] = np.nan        # 停牌：收盘价缺失（估值必须 ffill）
    open_["D"] = np.nan                     # 全程无价
    close["D"] = np.nan
    ok = pd.DataFrame(True, index=DAYS, columns=CODES)
    ok.loc[DAYS[1], "C"] = False            # 执行日 1：C 买不到
    ok.loc[DAYS[6], "B"] = False
    # T-2.3：engine 现在从 reason 派生 can_buy / can_sell。
    # 这里让 reason 与 can_trade 完全一致（True→0，False→1 停牌），
    # 于是 can_buy == can_sell == can_trade ⇒ 原有全部断言保持原意不变 ✓
    reason = pd.DataFrame(
        np.where(ok.to_numpy(), 0, 1).astype("int8"), index=DAYS, columns=CODES)
    W = pd.DataFrame(1.0 / 3.0, index=RDATES, columns=CODES)
    W["D"] = 0.0
    return W, {"hfq_open": open_, "hfq_close": close, "can_trade": ok, "reason": reason}


# ── 手算参照值 ────────────────────────────────────────────────
FEE1 = (2.0 / 3.0) * FB
CASH1 = 1.0 - 2.0 / 3.0 - FEE1
NAV1 = 1.0 - FEE1
VAL2 = (1.0 / 30.0) * 12.0 + (1.0 / 30.0) * 9.0          # 执行日 2 开盘估值持仓 = 0.7
NAV_PRE2 = CASH1 + VAL2
tA, tB, tC = NAV_PRE2 / 3.0 / 12.0, NAV_PRE2 / 3.0 / 9.0, NAV_PRE2 / 3.0 / 10.0
dA = tA - 1.0 / 30.0
BUY2 = (tB - 1.0 / 30.0) * 9.0 + tC * 10.0
SELL2 = -dA * 12.0
FEE2 = BUY2 * FB + SELL2 * FS
CASH2 = CASH1 + SELL2 - BUY2 - FEE2
NAV2 = CASH2 + (tA + tB + tC) * 10.0

W, P = make_inputs()
snap = (W.copy(), P["hfq_open"].copy(), P["hfq_close"].copy(), P["can_trade"].copy())
res = EN.run_layer(W, P, list(RDATES), slippage=0.0)      # 基础 fixture 关掉滑点 → 手算参照值干净
nav, cash, trades, shares = res["nav"], res["cash"], res["trades"], res["shares"]
print(f"fixture：{len(DAYS)} 个交易日 × {len(CODES)} 只 ｜ 调仓日 {len(RDATES)} 个 → 执行日 {DAYS[1].date()} / {DAYS[5].date()}")

print("\n── ① 交易日索引与首日 ──")
check(list(nav.index) == list(DAYS), "nav.index = 全部交易日")
check(near(nav.iloc[0], 1.0), f"首日全现金 nav = {nav.iloc[0]:.10f}")

print("\n── ② 只在执行日成交 ──")
check(list(trades.index) == [DAYS[1], DAYS[5]], "trades.index = [d1, d5]")

print("\n── ③ 买不到的票不成交（掩码 + open 缺失）──")
check(near(trades.loc[DAYS[1], "A"], 1.0 / 30.0, 1e-6), f"d1 A 成交股数变动 = {trades.loc[DAYS[1], 'A']:.8f}（1/30）")
check(near(trades.loc[DAYS[1], "B"], 1.0 / 30.0, 1e-6), f"d1 B 成交股数变动 = {trades.loc[DAYS[1], 'B']:.8f}（1/30）")
check(near(trades.loc[DAYS[1], "C"], 0.0, 1e-12), "d1 C 不可交易 → 成交 0（钱留在现金）")
check(near(cash.loc[DAYS[1]], CASH1), f"d1 现金 = {cash.loc[DAYS[1]]:.10f}")

print("\n── ④ 换手：一侧口径 0.5×(买+卖)/成交前净值 ──")
check(near(res["turnover"].loc[DAYS[1]], 1.0 / 3.0, 1e-12), f"d1 换手 = {res['turnover'].loc[DAYS[1]]:.10f}（1/3）")

print("\n── ⑤ 费用从现金扣、只扣一次 ──")
check(near(res["fees"].loc[DAYS[1]], FEE1, 1e-15), f"d1 费用 = {res['fees'].loc[DAYS[1]]:.12f}（(2/3)×fee_buy）")
check(near(nav.loc[DAYS[1]], NAV1), f"d1 净值 = {nav.loc[DAYS[1]]:.10f} = 1 − 费用")

print("\n── ⑥ 非调仓日：股数不变 ──")
check(all(near(nav.loc[d], NAV1) for d in DAYS[2:5]), "d2 = d3 = d4 = d1 净值（价格不变）")

print("\n── ⑦ 执行日 2：成交价用 open(12/9) 而不是 close(10) ──")
got = trades.loc[DAYS[5], ["A", "B", "C"]].to_numpy(dtype="float64")     # trades = 股数**变动**（delta）
exp = np.array([dA, tB - 1.0 / 30.0, tC])
# T-2.3：scale = 1/(1+fee_buy) 会让成交量整体压低 O(fee) 级（相对 ≈2.5e-5），
# 手算值是按「刚好买到目标」推的，所以差几十个 1e-6。本条的本意是抓
# 「用错价格」——12 vs 10 是 20% 的差异，1e-4 容差完全不影响它 ✓
check(np.allclose(got, exp, atol=1e-4), f"d5 成交股数变动 = {np.round(got, 8).tolist()}（手算 {np.round(exp, 8).tolist()}）")
check(got[0] < 0, "A（开盘 12 涨了）→ 卖出（若错用 close=10，会变成买入 +0.0011）")
check(got[1] > 0 and got[2] > 0, "B（开盘 9 跌了）→ 继续买入；C 首次买入")

print("\n── ⑧ 执行日 2：净值与账本恒等式 ──")
# 同上：scale 让净值差 O(fee²)+累积量级（实测 ~1.7e-6）
check(near(nav.loc[DAYS[5]], NAV2, 1e-5), f"d5 净值 = {nav.loc[DAYS[5]]:.10f}（手算 {NAV2:.10f}）")
# 同上：费用随成交额同比例变小
# 手算 FEE2 按「刚好买到目标」推；scale 让成交额小 2.5e-5 相对 ⇒ 费用同比例小 ~5e-8
check(near(res["fees"].loc[DAYS[5]], FEE2, 1e-6), f"d5 费用 = {res['fees'].loc[DAYS[5]]:.12f}")
val_px = P["hfq_close"].ffill()
okid = all(near(nav.loc[d], cash.loc[d] + float((shares.loc[d].to_numpy(dtype="float64") * val_px.loc[d].fillna(0.0).to_numpy()).sum()), 1e-6)
           for d in [DAYS[1], DAYS[5]])
check(okid, "nav = 现金 + Σ 股数 × 估值价")

print("\n── ⑨ 停牌日估值用 ffill（收盘价 NaN 不许归零）──")
check(near(nav.loc[DAYS[6]], nav.loc[DAYS[5]], 1e-9), f"d6 停牌（B 收盘 NaN）净值不动：{nav.loc[DAYS[6]]:.10f}")

print("\n── ⑪ 未来函数：改 d6/d7 价格不影响过去 ──")
W2, P2 = make_inputs(scale_tail=5.0)
res2 = EN.run_layer(W2, P2, list(RDATES), slippage=0.0)
check(np.allclose(res2["nav"].to_numpy()[:6], nav.to_numpy()[:6], atol=1e-12), "d0–d5 净值逐格不变")
check(not np.allclose(res2["nav"].to_numpy()[6:], nav.to_numpy()[6:]), "（对照）d6/d7 确实变了 → 说明缩放生效")

print("\n── ⑫ 入参未被改动 ──")
check(W.equals(snap[0]) and P["hfq_open"].equals(snap[1]) and P["hfq_close"].equals(snap[2]) and P["can_trade"].equals(snap[3]),
      "W / hfq_open / hfq_close / can_trade 逐格一致")

print("\n── ⑬ 多层与空层 ──")
res_all = EN.run_all_layers([W] * 5, P, list(RDATES), slippage=0.0)
nf = EN.nav_frame(res_all)
check(nf.shape == (len(DAYS), 5), f"nav_frame 形状 = {nf.shape}")
check(np.allclose(nf.to_numpy(), np.repeat(nav.to_numpy()[:, None], 5, axis=1), atol=1e-12), "五层各自独立、同权重 → 净值相同")
W0 = W.copy()
W0.iloc[:, :] = 0.0
res0 = EN.run_layer(W0, P, list(RDATES), slippage=0.0)
check(near(res0["nav"].iloc[-1], 1.0) and bool((res0["trades"].to_numpy() == 0).all()), "全 0 权重层：净值恒 1、零成交")

print("\n── ⑭ 全程无价的票 ──")
check(bool((trades["D"].to_numpy() == 0).all()), "D 从未成交")
check(bool((shares["D"].to_numpy() == 0).all()), "D 持仓恒 0")

print("\n── ⑮ 滑点：买价更贵、卖价更便宜（单边 5 bp）──")
res_s = EN.run_layer(W, P, list(RDATES), slippage=0.0005)
d_nav = float(nav.iloc[-1] - res_s["nav"].iloc[-1])
check(res_s["nav"].iloc[-1] < nav.iloc[-1], f"5bp 滑点后净值更低：{res_s['nav'].iloc[-1]:.10f} < {nav.iloc[-1]:.10f}")
check(d_nav > (2.0 / 3.0) * 0.0005 * 0.5, f"净值差 {d_nav:.8f} 与'成交额 × 滑点'同量级（首笔买入额 2/3）")
check(near(res_s["turnover"].loc[DAYS[1]], 0.5 * (2.0 / 3.0) * 1.0005, 1e-12),
      "换手按**含滑点**的成交金额算")
check(np.allclose(res_s["trades"].loc[DAYS[1]].to_numpy(), trades.loc[DAYS[1]].to_numpy(), atol=1e-9),
      "d1 股数按盘口价折算：同一 nav_pre 下股数不变（滑点只改现金）")

print("\n" + "=" * 46)
print("\n── ⑯ 成交前估值兜底：不许用【当天收盘】（未来函数）──")
D2 = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"])
close2 = pd.DataFrame({"R": [10.0, 10.0, 10.0, 20.0, 20.0], "S": [10.0] * 5}, index=D2)
open2 = pd.DataFrame({"R": [10.0, 10.0, 10.0, np.nan, 10.0], "S": [10.0] * 5}, index=D2)
ok2 = pd.DataFrame(True, index=D2, columns=["R", "S"])
W2b = pd.DataFrame({"R": [0.5, 0.5], "S": [0.5, 0.5]}, index=[D2[0], D2[2]])
r2 = EN.run_layer(W2b, {"hfq_open": open2, "hfq_close": close2, "can_trade": ok2,
                        "reason": pd.DataFrame(np.where(ok2.to_numpy(), 0, 1).astype("int8"),
                                               index=ok2.index, columns=ok2.columns)}, [D2[0], D2[2]], slippage=0.0)
# d1 建仓（各 0.05 股 @10 → 现金 = −0.00025）；d3：R 开盘缺失（不可成交）但**当天收盘 = 20**
nav_pre_fix = -0.00025 + 0.05 * 10.0 + 0.05 * 10.0      # 兜底用"上一交易日收盘"（= 10）✅
nav_pre_bad = -0.00025 + 0.05 * 20.0 + 0.05 * 10.0      # 若错用"当天收盘"（= 20）❌
tgt_s_fix = nav_pre_fix * 0.5 / 10.0
tgt_s_bad = nav_pre_bad * 0.5 / 10.0
got_s = float(r2["shares"].loc[D2[3], "S"])
check(abs(got_s - tgt_s_fix) < 1e-7, f"兜底用'上一交易日收盘'：S 目标股数 {got_s:.9f}（手算 {tgt_s_fix:.9f}）")
check(abs(got_s - tgt_s_bad) > 1e-3, f"（反例）若用'当天收盘 20' 会得到 {tgt_s_bad:.9f} —— 现在不是它")
# 建仓量 = 目标 0.05 × scale(1/(1+fee)) —— 精确值能算，容差保持 1e-8 ✓
check(near(float(r2["shares"].loc[D2[3], "R"]), 0.05 / (1.0 + 0.00025), 1e-8) and near(float(r2["trades"].loc[D2[3], "R"]), 0.0, 1e-12),
      "R 当天不可成交（开盘缺失）→ 保持原股数（建仓量已含 scale）、零成交")
print('\n── ⑰ 末位边界：最后一个调仓日没有下一交易日 ──')
RD_TAIL = [DAYS[0], DAYS[4], DAYS[7]]
W_tail = pd.DataFrame(1.0 / 3.0, index=RD_TAIL, columns=CODES)
W_tail['D'] = 0.0
res_tail = EN.run_layer(W_tail, P, RD_TAIL, slippage=0.0)
check(list(res_tail['trades'].index) == [DAYS[1], DAYS[5]], '末日调仓日被跳过（没有下一交易日 → 不凭空成交）')
check(bool(res_tail['nav'].notna().all()), '净值仍然逐日有值（不是只有执行日）')

print('\n── ⑱ 调仓日不在面板索引里（get_indexer = -1）──')
RD_BAD = [DAYS[0], pd.Timestamp('2020-01-04')]
W_bad = pd.DataFrame(1.0 / 3.0, index=RD_BAD, columns=CODES)
W_bad['D'] = 0.0
res_bad = EN.run_layer(W_bad, P, RD_BAD, slippage=0.0)
check(list(res_bad['trades'].index) == [DAYS[1]], '不在索引里的调仓日被跳过，且不崩')

print('\n── ⑲ 中途退市：按最后有效价永久计值（钉住已知偏差）──')
D3 = pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-11'])
close3 = pd.DataFrame({'X': [10.0, 10.0, np.nan, np.nan, np.nan, np.nan], 'Y': [10.0] * 6}, index=D3)
open3 = pd.DataFrame({'X': [10.0, 10.0, np.nan, np.nan, np.nan, np.nan], 'Y': [10.0] * 6}, index=D3)
ok3 = pd.DataFrame(True, index=D3, columns=['X', 'Y'])
W3 = pd.DataFrame({'X': [0.5], 'Y': [0.5]}, index=[D3[0]])
r3 = EN.run_layer(W3, {'hfq_open': open3, 'hfq_close': close3, 'can_trade': ok3,
                        'reason': pd.DataFrame(np.where(ok3.to_numpy(), 0, 1).astype("int8"),
                                               index=ok3.index, columns=ok3.columns)}, [D3[0]], slippage=0.0)
n3 = r3['nav'].to_numpy()
# T-2.3：scale = 1/(1+fee_buy)，所以净值 = 1/(1+fee)，和 1−fee 差 fee² 级
check(near(n3[1], 1.0 / (1.0 + 0.00025), 1e-12), f'建仓后净值 = 1/(1+费率)：{n3[1]:.12f}')
check(bool(np.allclose(n3[2:], n3[1], atol=1e-12)), '退市后 X 按最后有效价计值 → 净值恒定（偏乐观，已知偏差被钉住）')
# 同理：目标 0.05 股 × scale(1/(1+fee)) = 实际买入
check(near(float(r3['shares'].loc[D3[1], 'X']), 0.05 / (1.0 + 0.00025), 1e-8),
      'X 的股数保持不变（退市后不成交；建仓量已含 scale）')

print('\n── ⑳ W 的列与面板列不一致：顺序 / 子集 / 多余列 ──')
W_rev = W[['D', 'C', 'B', 'A']].copy()
res_rev = EN.run_layer(W_rev, P, list(RDATES), slippage=0.0)
check(near(float(res_rev['shares'].loc[DAYS[1], 'A']), float(shares.loc[DAYS[1], 'A']), 1e-7),
      '列顺序打乱 → 结果不变（按标签对齐，不是按位置）')
W_sub = W.drop(columns=['D'])
res_sub = EN.run_layer(W_sub, P, list(RDATES), slippage=0.0)
check(near(float(res_sub['shares'].loc[DAYS[1], 'A']), float(shares.loc[DAYS[1], 'A']), 1e-7),
      'W 是面板列的子集 → 只处理该子集')
raised = False
W_extra = W.copy()
W_extra['ZZZ'] = 0.0
try:
    EN.run_layer(W_extra, P, list(RDATES), slippage=0.0)
except KeyError:
    raised = True
check(raised, 'W 里有面板不存在的列 → KeyError（缺列要吵，不许静默）')

print('\n── ㉑ 现金约束：持仓卖不掉时不再能无成本融资（T-2.3 转正）──')
D4 = pd.DatetimeIndex(['2021-02-01', '2021-02-02', '2021-02-03', '2021-02-04', '2021-02-05'])
close4 = pd.DataFrame({'P': [10.0] * 5, 'Q': [10.0] * 5}, index=D4)
open4 = pd.DataFrame({'P': [10.0, 10.0, 10.0, np.nan, 10.0], 'Q': [10.0] * 5}, index=D4)
ok4 = pd.DataFrame(True, index=D4, columns=['P', 'Q'])
W4 = pd.DataFrame({'P': [1.0, 0.0], 'Q': [0.0, 1.0]}, index=[D4[0], D4[2]])
r4 = EN.run_layer(W4, {'hfq_open': open4, 'hfq_close': close4, 'can_trade': ok4,
                        'reason': pd.DataFrame(np.where(ok4.to_numpy(), 0, 1).astype("int8"),
                                               index=ok4.index, columns=ok4.columns)}, [D4[0], D4[2]], slippage=0.0)
cash4 = float(r4['cash'].loc[D4[3]])
nav4 = float(r4['nav'].loc[D4[3]])
hold4 = float(r4['shares'].loc[D4[3], 'P']) * 10 + float(r4['shares'].loc[D4[3], 'Q']) * 10
# T-2.3 前：P 卖不掉 ⇒ 拿不到钱 ⇒ 却仍按 100% 买 Q ⇒ cash = −0.9、杠杆 ≈ 2 ✗
# T-2.3 后：卖不掉 ⇒ 现金不增加 ⇒ scale ≈ 0 ⇒ Q 买不进 ⇒ cash ≥ 0、杠杆 ≤ 1 ✓
check(cash4 >= -1e-9, f'不再允许透支：cash = {cash4:.3e}（P 卖不掉 ⇒ 没钱买 Q）')
_scale4 = 1.0 / (1.0 + 0.00025)   # 建仓时 scale = 1/(1+fee_buy) ⇒ P 不是刚好 0.1 股
check(near(float(r4['shares'].loc[D4[3], 'P']), 0.1 * _scale4, 1e-8),
      f'P 股数 = 建仓所得 {0.1 * _scale4:.9f}（卖不掉，之后一格没动）')
check(hold4 / nav4 <= 1.001, f'杠杆 = 持仓市值/净值 = {hold4 / nav4:.4f}（满仓上限，不再有隐含融资）')
diff4 = nav4 - (cash4 + hold4)
check(near(nav4, cash4 + hold4, 1e-6),
      f'恒等式仍成立：nav − (现金 + 持仓市值) = {diff4:.3e}（shares 快照是 float32 → 容差 1e-6）')

print('\n── ㉒ init_cash 尺度不变性（钉住「目前无整手 / 无最低佣金」）──')
res_small = EN.run_layer(W, P, list(RDATES), slippage=0.0, init_cash=1.0)
res_big = EN.run_layer(W, P, list(RDATES), slippage=0.0, init_cash=1000000.0)
check(bool(np.allclose(res_big['nav'].to_numpy() / 1000000.0, res_small['nav'].to_numpy(), atol=1e-9)),
      '1.0 与 1e6 的净值曲线完全同形（分数股、无最低佣金 → 本金只是尺度）')

print('\n── ㉓ 诊断列：开盘净值 / 成交前净值（估值与交易同时点 → 账能对上）──')
nav_open_s = res['nav_open']
check(bool(nav_open_s.notna().all()) and list(nav_open_s.index) == list(DAYS),
      'nav_open 与 nav 同索引、逐日有值（估值不只在执行日发生）')
check(bool(np.allclose((res['nav_pre'] - res['fees']).to_numpy(),
                       nav_open_s.loc[res['nav_pre'].index].to_numpy(), atol=1e-9)),
      f"恒等式：nav_open(执行日) = nav_pre − 费用（d1 实测 {float(nav_open_s.loc[DAYS[1]]):.10f}）")

print('\n── ㉕ 被挡的调仓（T-2.4）：想买买不到 / 想卖卖不掉，都要留下来 ──')
bb = res["blocked_buy"]
bs = res["blocked_sell"]
check(bb.shape == (len(res["nav_pre"]), len(CODES)),
      f"blocked_buy 形状 = (执行日, 股票)：{bb.shape}")
# 基础 fixture：C 在 DAYS[1] can_trade=False（reason=1 停牌）⇒ 想买买不到
check(float(bb.loc[DAYS[1], "C"]) > 0.0,
      f"d1 想买 C 但买不到 ⇒ 记下 {float(bb.loc[DAYS[1], 'C']):.6f} 股")
check(float(bb.loc[DAYS[1], "A"]) == 0.0,
      "A 那天没被挡 ⇒ 一格都不许有值（有值就说明掩码用错了）")

check(passed + failed >= 15, f"断言条数守卫：{passed + failed} 条")
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print("有断言未通过 ❌")
    sys.exit(1)
print("全部通过 ✅ 账本 / 成交 / 费用 / 换手 可用（下一步：metrics）")