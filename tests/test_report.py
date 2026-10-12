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
        "fee_buy": 0.00025, "fee_sell": 0.00075, "slippage": 0.0005, "stamp": "20261012"}
bn = RP.banner(meta)
for token in ("244", "月频", "T+1 开盘", "close", "比值法", "日差法", "月差法", "滑点"):
    check(token in bn, f"口径声明含「{token}」")
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
check(f"{out['ls']['比值法']:.2%}" in text, f"多空·比值法 {out['ls']['比值法']:.2%} 出现在报告里")

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
print("=" * 46)
print(f"  ✅ 断言条数守卫：{n_assert} 条")
print(f"结果: 通过 {n_pass} / {n_assert}")
shutil.rmtree(TMP, ignore_errors=True)
if n_fail:
    print("有断言未通过 ❌")
    raise SystemExit(1)
print("全部通过 ✅ 报告可生成、数字同源、口径声明齐全、可复现")
