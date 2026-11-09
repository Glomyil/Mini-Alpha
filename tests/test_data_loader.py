# -*- coding: utf-8 -*-
"""
loader.py 验收测试（数据层 / S0）

覆盖三类分支：
  ① 候选清单        ② 主路径（读→守卫→解析→裁剪→reset_index）
  ③ 边界（首尾日含入） ④ 行空分支（区间外 / 退市股 / 不存在 → None）
  ⑤ 生成器逐只产出（区间内无数据的会被跳过）

用法（项目根目录任意终端）:
  python "tests/test_data_loader.py"
期望输出末尾: 全部通过 (X/Y)

说明：本文件是"被直接运行的脚本"，所以开头必须先修 sys.path
      （import 是运行时语句，路径修正必须写在 import 之前）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径准备：保证能从项目根 import data.loader
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from data.loader import iter_frames, list_codes, load_one

REQUIRED = ["date", "open", "high", "low", "close", "volume"]

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


def check_eq(actual, expected, label):
    check(actual == expected, f"{label}: {actual}" + ("" if actual == expected else f" (期望 {expected})"))


# ---------------------------------------------------------------------------
# ① 候选清单
# ---------------------------------------------------------------------------
print("── ① 候选清单 ──")
codes = list_codes()
check(len(codes) > 5000, f"清单数量 {len(codes)} > 5000")
check(len(codes) == len(set(codes)), "清单内无重复代码")

# ---------------------------------------------------------------------------
# ② 主路径：真实代码 + 合法区间（走完全部代码行）
# ---------------------------------------------------------------------------
print("\n── ② 主路径 ──")
df = load_one("sh.600000", "2020-01-01", "2020-12-31")
check(df is not None, "返回 DataFrame（不是 None）")
if df is not None:
    # 2026-11-04（T-6.1）：抓取层加了 turn 列 ⇒ 11 → 12
    #   （10 个 KLINE 字段 + turn + back_adj_factor；turn 是流通市值的原料）
    check_eq(df.shape, (243, 12), "shape")
    check("turn" in df.columns, "turn 列存在（T-6.1：流通市值 = close × volume/(turn/100)）")
    check(all(c in df.columns for c in REQUIRED), f"必需列齐全 {REQUIRED}")
    check(list(df.index) == list(range(len(df))), "索引从 0 连续（reset_index 生效）")

    # -----------------------------------------------------------------------
    # ③ 边界：区间首尾日必须被包含
    # -----------------------------------------------------------------------
    print("\n── ③ 边界（含首尾）──")
    check(df["date"].min() >= pd.Timestamp("2020-01-01"), "首日 >= 区间开始")
    check(df["date"].max() <= pd.Timestamp("2020-12-31"), "末日 <= 区间结束")
    check_eq(str(df["date"].iloc[0].date()), "2020-01-02", "首行 = 区间内第一个交易日（1/1 休市）")
    check_eq(str(df["date"].iloc[-1].date()), "2020-12-31", "末行 = 区间最后一个交易日")
    check(bool(df["date"].is_monotonic_increasing), "date 升序")
    check(not bool(df["date"].duplicated().any()), "date 无重复")

# ---------------------------------------------------------------------------
# ④ 行空分支：都是"业务现象"，应为 None（不是报错）
# ---------------------------------------------------------------------------
print("\n── ④ 行空分支 → None ──")
check(load_one("sh.600000", "2030-01-01", "2030-12-31") is None, "区间外 → None")
check(load_one("sh.600005", "2020-01-01", "2020-12-31") is None, "退市股（2017-02 退市）→ None")
check(load_one("sh.999999", "2020-01-01", "2020-12-31") is None, "不存在的代码 → None")

# ---------------------------------------------------------------------------
# ⑤ 生成器：逐只产出；区间内无数据的被跳过（产出数 ≤ 输入数，别断言相等）
# ---------------------------------------------------------------------------
print("\n── ⑤ 生成器 iter_frames ──")
sample = codes[:5]
got = list(iter_frames(sample, "2020-01-01", "2020-12-31"))
check(1 <= len(got) <= len(sample), f"产出 {len(got)} 只（≤ 输入 {len(sample)} 只：退市股被跳过）")
check(all(code in sample for code, _ in got), "产出的 code 都来自输入清单")
check(all(isinstance(d, pd.DataFrame) and len(d) > 0 for _, d in got), "每只产出都非空")

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ loader 数据层可用（列齐 + 区间内 + 有行）")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
