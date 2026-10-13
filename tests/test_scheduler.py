# -*- coding: utf-8 -*-
"""test_scheduler.py —— 总调度（v0）的验收

（注意：**不是** test_pipeline.py —— 那份测的是清洗层的 data/cleaner/pipeline.py）

分两层：
  A 结构检查：**静态读** main.py（不执行它）—— 口径单源 / 无指标公式 / 不吞异常 / 软失败可见
  B 功能检查：用真数据把主干链跑一遍（panels → rdates → 软失败 → weights → engine → nav_frame → report）
              产物落到项目根 _tmp_pipeline_test/（跑完自动删）

跑法：python tests/test_scheduler.py     期望：全部通过
"""
from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.panel import load_panels                          # noqa: E402
from backtest.assemble import build_features, rebalance_dates    # noqa: E402
from backtest.weights import weights_for_factor                  # noqa: E402
from backtest.engine import run_all_layers, nav_frame            # noqa: E402
from backtest.factors import FACTORS                             # noqa: E402
from backtest import report as RP                                # noqa: E402

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


MAIN = ROOT / "main.py"
TMP = ROOT / "_tmp_pipeline_test"
SINCE = 2016
FACTOR = "rev_20d"

print("── A. 结构检查（静态读 main.py，不执行）──")
if not MAIN.exists():
    print("  ❌ 找不到 main.py")
    raise SystemExit(1)
src = MAIN.read_text(encoding="utf-8")
code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]

n_2016 = sum(1 for l in code_lines if "2016" in l)
check(n_2016 == 1, f"口径单源：`2016` 在 main.py 代码行中只出现 1 次（实测 {n_2016}）")
n_filter = len(re.findall(r"year\s*>=", src))
check(n_filter <= 1, f"`year >=` 不超过 1 处（实测 {n_filter}）")

forbidden = ["pct_change", "cummax", "np.sqrt", ".std(", "cumprod"]
hit = [t for t in forbidden if t in src]
check(not hit, f"main.py 里没有指标公式算子（检查 {forbidden}；命中 {hit}）")
check(len(re.findall(r"except[^\n]*:\s*\n\s*pass", src)) == 0,
      "没有 `except: pass`（异常不许被吞）")
check("requires" in src and src.count("continue") >= 1,
      "有 requires 前置检查 + continue（软失败要可见、要跳过）")

tree = ast.parse(src)
mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
check(len(mains) == 1, f"只允许一个 `def main`（实测 {len(mains)} 个 —— 多定义会静默覆盖，程序跑的是最后一个）")
if len(mains) == 1:
    has_lp = any(isinstance(x, ast.Call) and getattr(x.func, "id", "") == "load_panels"
                 for x in ast.walk(mains[0]))
    check(has_lp, "`main()` 就是调度入口（它内部调用 load_panels）")

lp = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_panels"]
fors = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
check(len(lp) == 1 and fors and lp[0].lineno < min(n.lineno for n in fors),
      "面板只加载一次，且在因子循环之外（缓存不重复加载）")

print("── B. 功能检查（真数据，约 15 秒）──")
panels = load_panels()
check(sorted(panels) == ["can_trade", "hfq_close", "hfq_open", "ret_1d_hfq"],
      f"面板 4 张表：{sorted(panels)}")
check(panels["hfq_close"].shape == (2839, 5471), f"面板形状 {panels['hfq_close'].shape}")

rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= SINCE]
check(len(rdates) == 129, f"调仓日 {len(rdates)} 个（期望 129）")
check(all(d.year >= SINCE for d in rdates), "全部 ≥ 2016（含 2016 本身，不是 >）")
check(all(d in panels["hfq_close"].index for d in rdates), "调仓日都是面板里的真实交易日")

feats_amihud = build_features(panels, ["amihud_20d"])
check(feats_amihud == {}, "软失败：amihud_20d 缺 amount → 返回空 dict（不抛异常）")
check([t for t in FACTORS["amihud_20d"].get("requires", []) if t not in panels] == ["amount"],
      "缺的表就是 ['amount']（调度里会被记成 skipped）")

W, layers = weights_for_factor(FACTOR, panels, rdates)
check(isinstance(W, list) and len(W) == 5, f"W 是 list[5]（实测 {type(W).__name__}/{len(W)}）")
check(layers.shape == (129, 5471), f"layers 形状 {layers.shape}")
rowsum = np.round(W[0].sum(axis=1).to_numpy(), 6)
check(bool(np.isin(rowsum, [0.0, 1.0]).all()), "W 每行权重和 ∈ {0.0, 1.0}（满仓或空层）")
check(int(layers.to_numpy().min()) == -1 and int(layers.to_numpy().max()) == 4,
      "层号值域 = {−1, 0..4}（−1 = 未入层）")

res = run_all_layers(W, panels, rdates)
check(isinstance(res, list) and len(res) == 5, "res 是 list[5]")
check(set(res[0]) == {"nav", "nav_open", "nav_pre", "cash", "turnover", "fees", "trades", "shares"},
      "每层 8 本账")
check(abs(float(res[0]["nav"].iloc[0]) - 1.0) < 1e-12, "净值从 1.0 起步（每层独立账户）")

nav_df = nav_frame(res)
check(nav_df.shape == (2839, 5) and list(nav_df.columns) == [0, 1, 2, 3, 4],
      f"nav_df {nav_df.shape}，列 = 层号 0..4")

shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)
out = RP.build_report(FACTOR, nav_df, panels, rdates, res, outdir=TMP)
text = out["md"].read_text(encoding="utf-8")
check(out["md"].exists() and out["png"].exists(), f"报告产物：{out['md'].name} + {out['png'].name}")
check("口径声明" in text and "244" in text, "报告含口径声明（且年化常数是 244）")
v_ann = float(out["layer_tab"].loc[2, "年化"])
check(f"{v_ann:.2%}" in text, f"同源：层 2 年化 {v_ann:.2%} 出现在报告里")

print("── C. 确定性 & 输入不可变（抓「能跑但错」的幽灵 bug）──")
healthy = float(panels["can_trade"].to_numpy().mean())
check(0.5 < healthy < 0.95, f"can_trade 健康度：True 占比 {healthy:.1%}（退化成全 False 会让所有交易被拒）")
open_valid = float(panels["hfq_open"].notna().to_numpy().mean())
check(open_valid > 0.5, f"hfq_open 非空占比 {open_valid:.1%}（退化成全 NaN 会让引擎买不进任何票）")

before_fp = {k: int(pd.util.hash_pandas_object(v.fillna(0)).sum()) for k, v in panels.items()}
res2 = run_all_layers(W, panels, rdates)
nav2 = nav_frame(res2)
check(all(before_fp[k] == int(pd.util.hash_pandas_object(v.fillna(0)).sum()) for k, v in panels.items()),
      "引擎不改输入：跑完 engine 后四张面板指纹不变")
same = np.array_equal(res[0]["nav"].to_numpy(), res2[0]["nav"].to_numpy(), equal_nan=True) and \
    all(np.array_equal(res[k]["nav"].to_numpy(), res2[k]["nav"].to_numpy(), equal_nan=True) for k in range(5))
check(same, "确定性：同一因子连跑两次 → 五层净值逐位相同")

print()
print("=" * 46)
print(f"  ✅ 断言条数守卫：{n_assert} 条")
print(f"结果: 通过 {n_pass} / {n_assert}")
shutil.rmtree(TMP, ignore_errors=True)
if n_fail:
    print("有断言未通过 ❌")
    raise SystemExit(1)
print("全部通过 ✅ 主干链可跑 + 主干不重算指标 + 口径单源 + 软失败未被吞")
