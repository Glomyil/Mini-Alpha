# -*- coding: utf-8 -*-
"""test_report.py —— report.py 的验收（§6 的 12 条清单）

跑法：python tests/test_report.py
期望：全部通过（条数守卫会核对断言数量，防测试空转）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from backtest import report as RP
except ImportError:
    print("❌ 还没写 backtest/report.py（或名字不对：应当是 report.py，不是 reports.py）")
    print("   先照《report.py 教学参考》§7 写，再跑本测试")
    raise SystemExit(1)

from backtest import metrics as MT   # noqa: E402

n_pass = 0
n_fail = 0
n_assert = 0


def check(ok, msg):
    global n_pass, n_fail, n_assert
    n_assert += 1
    if ok:
        n_pass += 1
        print(f"  ✅ {msg}")
    else:
        n_fail += 1
        print(f"  ❌ {msg}")


idx = pd.date_range("2020-01-01", periods=40, freq="B")
nav_df = pd.DataFrame({
    0: np.linspace(1.0, 0.90, 40),
    1: np.linspace(1.0, 1.00, 40),
    2: np.linspace(1.0, 1.10, 40),
    4: np.linspace(1.0, 1.30, 40),
}, index=idx)
rdates = [idx[0], idx[10], idx[20], idx[30], idx[-1]]

TMP = Path(__file__).resolve().parent.parent / "_tmp_report_test"   # 落在工作区根（沙箱只允许写仓库内）
TMP.mkdir(parents=True, exist_ok=True)

print("── ① md_table：表头 / 行数 / 比率列转百分比 ──")
tab = MT.layer_table(nav_df)
md = RP.md_table(tab)
lines = [ln for ln in md.splitlines() if ln.strip()]
check(len(lines) >= 2 + len(nav_df.columns), f"表头+分隔+{len(nav_df.columns)} 行 = 至少 {2 + len(nav_df.columns)} 行")
check("期末净值" in md and "夏普" in md, "表头含列名（期末净值 / 夏普）")
check("%" in md, "比率列含百分比符号（口径⑦）")
ann_cell = [c.strip() for c in lines[2].split("|") if c.strip()][2]
check(ann_cell.endswith("%"), f"年化列是百分比（实测单元格 = {ann_cell}）")
sharpe_cell = [c.strip() for c in lines[2].split("|") if c.strip()][-1]
check(not sharpe_cell.endswith("%"), f"夏普是纯数、不许转百分比（实测 = {sharpe_cell}）")

print("── ② banner：口径声明必须齐全 ──")
meta = {"factor": "rev_20d", "start": idx[0].date(), "end": idx[-1].date(), "n_rdate": len(rdates),
        "trading_days": 244, "periods_per_year": 12, "val_basis": "close",
        "freq": "ME", "universe": "all",
        "fee_buy": 0.00025, "fee_sell": 0.00075, "slippage": 0.0005, "stamp": "20261012"}
bn = RP.banner(meta)
# 2026-11-03：多空口径改了（比值法/日差法/月差法 → 标准复利 + 层差 pp + 扣融券净额）。
# 这里换成新口径的稳定标记 —— 不再断言已删除的三种旧算法名。
for token in ("244", "T+1 开盘", "close", "滑点", "标准复利", "纯度诊断"):
    check(token in bn, f"口径声明含「{token}」")

# ★ 2026-11-03 新发现的误导性 bug：调仓频率写死成「月频」。
#   证据：改 freq="W-FRI" 后报告仍印「调仓：月频（553 期）」—— 月频 553 期 = 46 年。
#   判据不能是「含月频」（那正好是 bug 本身），必须是【跟着 meta["freq"] 变】。
check("月频" in bn and "12 期/年" in bn, "freq=ME ⇒ 月频 / 12 期/年")
for _code, _label in (("QE", "季频"), ("W-FRI", "周频")):
    _m = dict(meta, freq=_code)
    _bn = RP.banner(_m)
    check(_label in _bn, f"freq={_code} ⇒ 口径声明写「{_label}」（实测 {_bn.splitlines()[1][:70]}）")
    check("月频" not in _bn, f"freq={_code} ⇒ 不许再出现「月频」")
# 未知频率不许静默退回「月频」—— 原样印出代码，让写错的人看得见（PITFALLS #19）
_bnX = RP.banner(dict(meta, freq="XX"))
check("XX" in _bnX and "月频" not in _bnX, "未知 freq ⇒ 原样印代码，不静默退回月频")
check("收盘估值" not in bn, "口径声明不再硬编码「收盘估值」（改为读 meta['val_basis']）")

print("── ③ build_report：一站式出报告（无 panels / 无 res）──")
out = RP.build_report("rev_20d", nav_df, None, rdates, None, outdir=TMP)
check(out["md"].exists(), f"md 已生成：{out['md'].name}")
check(out["png"].exists() and out["png"].stat().st_size > 10_000,
      f"png 已生成且 > 10 KB（实测 {out['png'].stat().st_size / 1024:.0f} KB）")
text = out["md"].read_text(encoding="utf-8")

print("── ④ 同源：报告里的数字 = metrics 算的 ──")
for L in nav_df.columns:
    v = out["layer_tab"].loc[L, "期末净值"]
    check(f"{v:.4f}" in text, f"层 {L} 期末净值 {v:.4f} 出现在报告里")
v_ann = out["layer_tab"].loc[2, "年化"]
check(f"{v_ann:.2%}" in text, f"年化按百分比渲染（{v_ann:.2%}）")
v_cum = out["ls"]["多空累计"]
# §5 的表格里累计那格用 :+.2% 渲染（带正负号），所以这里也跟着带号
check(f"{v_cum:+.2%}" in text, f"多空累计 {v_cum:+.2%} 出现在报告里")
check("比值法" not in text, "报告里不该再出现「比值法」（T-3.4 删掉的误导口径）")

print("── ⑤ 可复现：同一输入跑两次 → 文本逐字节相同 ──")
out2 = RP.build_report("rev_20d", nav_df, None, rdates, None, outdir=TMP)
check(out2["md"].read_text(encoding="utf-8") == text, "两次生成的 md 完全一致")

print("── ⑥ res=None 时不该出现换手/费用列 ──")
check("累计换手" not in text and "费用合计" not in text, "无 res 时报告不含换手/费用两列")

print("── ⑦ nav 含 NaN → 报错（透传 metrics 守卫，别在 report 里悄悄填）──")
nav_bad = nav_df.copy()
nav_bad.iloc[3, 0] = np.nan
try:
    RP.build_report("rev_20d", nav_bad, None, rdates, None, outdir=TMP)
    check(False, "含 NaN 的 nav 应当报错，但没报")
except ValueError:
    check(True, "含 NaN 的 nav → ValueError ✓")

print("── ⑧ 文件名含因子名与日期 ──")
check("rev_20d" in out["md"].name and out["md"].name.startswith("report_"), f"文件名 = {out['md'].name}")
check("close" in out["md"].name, "文件名含口径（val_basis）")
check(out["png"].name.startswith("nav_rev_20d"), f"图名 = {out['png'].name}")

print("── ⑨ 没有未替换的占位符 ──")
check("{" not in text and "}" not in text, "报告里没有未替换的 { } 占位符")

print("── ⑩ 结构性检查：report 不许自己写指标公式 ──")
src = (Path(RP.__file__)).read_text(encoding="utf-8")
forbidden = ["pct_change", "cummax", "np.sqrt", ".std(", "** (", "cumprod"]
hit = [t for t in forbidden if t in src]
check(not hit, f"report.py 里没有指标公式算子（检查 {forbidden}；命中 {hit}）")

print("── ⑪ 中文字体已显式设置 ──")
fonts = list(RP.plt.rcParams["font.sans-serif"])
check(any(("YaHei" in f) or ("SimHei" in f) for f in fonts), f"font.sans-serif = {fonts[:2]}")
check(RP.plt.rcParams["axes.unicode_minus"] is False, "axes.unicode_minus = False（负号不丢）")

print()
print('\n── ⑨ 有 res 时：首屏出现「## 0. 成交失败」（T-2.4）──')
# res 是 list[dict]（每层一个账本）；用层 4 => [_one] * 5 ✓
_ex = pd.DatetimeIndex([rdates[1], rdates[2]])
_flat = pd.DataFrame([[10.0], [11.0]], index=_ex, columns=[0])
_one = {
    "turnover":     pd.Series([0.5, 0.5], index=_ex),
    "fees":         pd.Series([0.001, 0.001], index=_ex),
    "blocked_buy":  pd.DataFrame([[0.01], [0.0]], index=_ex, columns=[0]),
    "blocked_sell": pd.DataFrame([[0.0], [0.0]], index=_ex, columns=[0]),
    "nav_pre":      pd.Series([1.0, 1.0], index=_ex),
}
_panels = {"hfq_close": _flat, "hfq_open": _flat,
           "can_trade": pd.DataFrame([[True], [True]], index=_ex, columns=[0]),
           "reason": pd.DataFrame([[2], [0]], index=_ex, columns=[0], dtype="int8")}
# factor_val 的列必须和 panels["hfq_close"] 的列一致（rank_ic 有断言）；
# 而且不能用全 0 —— 常数横截面会让 Spearman 算不出来、直接抛断言 ✗
_fv = pd.DataFrame(np.random.RandomState(0).randn(len(idx), 1),
                   index=idx, columns=[0])
out3 = RP.build_report("rev_20d", nav_df, _panels, rdates,
                       [_one] * 5, outdir=TMP, factor_val=_fv)
check("## 0. 成交失败" in out3["md"].read_text(encoding="utf-8"),
      "有 res 时报告首屏出现「## 0. 成交失败」")

print("── ⑫ 2026-11-04 新增字段：股票池 / IC胜率 / 多空夏普 / 多空回撤 / 每期换手 ──")
BN_SPEC = RP.banner(dict(meta, universe="all"))      # 明确给 all
text3 = out3["md"].read_text(encoding="utf-8")

# ① banner 的「股票池」——术语从「宇宙」改成券商研报的「股票池」
#    不传 spec 时 meta["universe"] 是 "?"，此时 banner 原样印 "?"
check("股票池：" in text3, "banner 含「股票池」一行（不是「宇宙」）")
check("宇宙" not in BN_SPEC, "banner 不出现「宇宙」这个硬译词")

# ② §4 的 IC>0 胜率
check("IC > 0 胜率" in text3, "§4 印出 IC>0 胜率")

# ③ §5 的多空夏普 / 最大回撤
# ⚠️ 不能用「最大回撤」判 §5 —— 它是 §1 表头的固有字（恒真 = 假绿，PITFALLS #8）。
# 用 §5 独有的整行片段来判。
check("多空夏普 **" in text3, "§5 印出多空夏普（带粗体标记，§1 没有这个字串）")
check("零成本口径，按" in text3, "§5 的最大回撤那行在（「零成本口径，按 N 期」只有 §5 有）")

# ④ §1 的「每期换手」列（累计换手 ÷ 期数，避免被读成"每期换手 105%"）
check("每期换手" in text3, "§1 有「每期换手」列")
_tab3 = out3["layer_tab"]
check("每期换手" in _tab3.columns, "layer_tab 带「每期换手」")
# 手算：out3 的 turnover 是 [0.5, 0.5] 两期 ⇒ 累计 1.0、每期 0.5
check(abs(float(_tab3.loc[0, "每期换手"]) - 0.5) <= 1e-12,
      f"每期换手 = 1.0/2 = 0.5（实测 {_tab3.loc[0, '每期换手']}）")

# ⑤ 不传 freq/universe 键时 banner 该 KeyError（docstring：缺一项就算没做这一格），
#    不许静默兜底 —— 所以 BN_NOSPEC 那条改成断言会抛异常
_raised = False
try:
    RP.banner({k: v for k, v in meta.items() if k not in ("freq", "universe")})
except KeyError:
    _raised = True
check(_raised, "banner 缺 freq/universe 键 ⇒ KeyError（不静默兜底，PITFALLS #19）")

# ⑥ ★ 接线测试：走真正的 build_report(..., spec=...)，验证 spec.freq → meta → banner
#    和 spec.periods_per_year → ic_summary。之前全仓【一处都没有】，写死回"月频"也能全绿。
class _FakeSpec:
    def __init__(self, freq, ppy):
        self.freq = freq
        self.periods_per_year = ppy
        self.trading_days = 244
        self.fee_buy = 0.00025
        self.fee_sell = 0.00075
        self.slippage = 0.0005
        self.borrow_cost = 0.08
        self.universe = "all"

# IC 要能真算出来，横截面必须 ≥ rank_ic 的 min_names(30)
# ⇒ 造一个 40 只票的 fixture（价格按列各不相同，否则常数横截面会让 Spearman 抛断言）
_N60 = 40
_rng60 = np.random.RandomState(7)
_px60 = 10.0 * np.cumprod(1.0 + _rng60.normal(0.0005, 0.02, (len(idx), _N60)), axis=0)
_CL60 = pd.DataFrame(_px60, index=idx, columns=list(range(_N60)))
_pan60 = {"hfq_close": _CL60, "hfq_open": _CL60,
          "can_trade": pd.DataFrame(True, index=idx, columns=list(range(_N60))),
          "reason": pd.DataFrame(0, index=idx, columns=list(range(_N60)), dtype="int8")}
_fv60 = pd.DataFrame(_rng60.randn(len(idx), _N60), index=idx, columns=list(range(_N60)))
_nav60 = pd.DataFrame({0: np.linspace(1.0, 0.9, len(idx)),
                       4: np.linspace(1.0, 1.3, len(idx))}, index=idx)

_ir_by_ppy = {}
for _code, _label, _ppy in (("QE", "季频", 4), ("W-FRI", "周频", 52)):
    _o = RP.build_report("rev_20d", _nav60, _pan60, rdates, None,
                         outdir=TMP, factor_val=_fv60, spec=_FakeSpec(_code, _ppy))
    _t = _o["md"].read_text(encoding="utf-8")
    check(f"调仓：{_label}" in _t, f"spec.freq={_code} ⇒ 报告 banner 写「{_label}」")
    check("月频" not in _t, f"spec.freq={_code} ⇒ 报告里不许出现「月频」")
    check(f"IC-IR 用 {_ppy} 期/年" in _t, f"spec.freq={_code} ⇒ banner 印 IC-IR 用 {_ppy} 期/年")
    check("全 A（含 ST" in _t, "spec.universe=all ⇒ 报告印出股票池正文（不是只印 ?）")
    # ★ 关键：IC-IR 的数值必须真按 √ppy 算 —— 只验 banner 上的字等于没验（PITFALLS #29）
    _m = MT.ic_summary(_o["ic"] if "ic" in _o else MT.rank_ic(_fv60, _pan60, rdates),
                       periods_per_year=_ppy)
    _exp = _m["ic_mean"] / _m["ic_std"] * (_ppy ** 0.5)
    _ir_by_ppy[_ppy] = _m["ic_ir"]
    check(np.isfinite(_m["ic_ir"]) and abs(_m["ic_ir"] - _exp) < 1e-9,
          f"IC-IR={_m['ic_ir']:.4f} 必须等于 mean/sd×√{_ppy}={_exp:.4f}")

# 两个 ppy 的 IC-IR 之比必须 = √(52/4) = √13 —— 证明 ppy 真的传进去了，不是都被忽略
check(abs(_ir_by_ppy[52] / _ir_by_ppy[4] - (52 / 4) ** 0.5) < 1e-9,
      f"IC-IR(52)/IC-IR(4) 必须 = √13 = {(52/4)**0.5:.4f}"
      f"（实测 {_ir_by_ppy[52]/_ir_by_ppy[4]:.4f}）")

print("=" * 46)
print(f"  ✅ 断言条数守卫：{n_assert} 条")
print(f"结果: 通过 {n_pass} / {n_assert}")
shutil.rmtree(TMP, ignore_errors=True)
if n_fail:
    print("有断言未通过 ❌")
    raise SystemExit(1)
print("全部通过 ✅ 报告可生成、数字同源、口径声明齐全、可复现")
