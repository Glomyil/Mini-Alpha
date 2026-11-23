# -*- coding: utf-8 -*-
"""
pipeline.py 冒烟测试（执行器：清单 → 叫号 → 报告）

覆盖：
  ① 空跑（0 环节）：loader 逐只交货，run_one 原样返回
  ② 玩具环节：df 真的在流动 + 报告记录新增列 + 白名单提醒生效
  ③ 环节忘了 return df → 必须是清晰的 TypeError（不是 NoneType 下标错）
  ④ ctx 契约：缺 asset_type / code → 当场 KeyError

用法（项目根目录）:
  python "tests/test_pipeline.py"
期望末尾: 全部通过 (X/Y)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import loader
from data.cleaner import pipeline

START, END = "2020-01-01", "2020-12-31"

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


# 玩具环节：加一列 toy（故意不在白名单里，用来验证提醒）
def toy_run(df, params, ctx, kind="stock"):   # kind 是 run() 契约的一部分（2026-11-18 统一）
    out = df.copy()
    out["toy"] = 1
    return out


TOY = type("ToyModule", (), {"run": staticmethod(toy_run)})


def bad_run(df, params, ctx):
    return None                      # 模拟"忘了 return df"


BAD = type("BadModule", (), {"run": staticmethod(bad_run)})

CTX_OK = {"universe": "stock", "asset_type": "stock", "code": "sh.600000"}


print("── ① 空跑（0 环节）──")
codes = loader.list_codes()[:3]
first_two = codes[:2]                # 第 3 只（sh.600005）是退市股，区间内无数据
n = 0
for code, df in loader.iter_frames(first_two, START, END):
    out, reps = pipeline.run_one(df, {**CTX_OK, "code": code}, steps=[])
    n += 1
    check(len(out) == len(df) and reps == [], f"{code}: 原样返回（{len(df)} 行，0 份报告）")
check(n == 2, f"共处理 {n} 只有数据的股票")

print("\n── ② 玩具环节：df 流动 + 报告 + 白名单提醒 ──")
df = loader.load_one("sh.600000", START, END)
out, reps = pipeline.run_one(df, CTX_OK, steps=[("toy", TOY)])
check(len(out) == len(df), "行数不变")
check("toy" in out.columns, "df 真的流动了（返回表里出现新列 toy）")
check(reps[0]["columns_added"] == ["toy"], f"报告记录新增列: {reps[0]['columns_added']}")
check(reps[0]["status"] == "has_warnings", "未登记列 → 报告 status 变成 has_warnings")

print("\n── ③ 环节忘了 return df → 清晰的 TypeError ──")
try:
    pipeline.run_one(df, CTX_OK, steps=[("bad", BAD)])
except TypeError as e:
    check("忘了 return" in str(e), f"报错清晰: {e}")
else:
    check(False, "环节返回 None 竟然没报错")

print("\n── ④ ctx 契约：缺键当场报错 ──")
try:
    pipeline.run_one(df, {"universe": "stock"})
except KeyError as e:
    check("asset_type" in str(e), f"缺键报错: {e}")
else:
    check(False, "ctx 缺必填键竟然没报错")

print("\n── ⑤ run_pipeline 端到端（0 环节，打印两行）──")
pipeline.run_pipeline(first_two, START, END, steps=[])

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ pipeline 执行器可用")
else:
    print(f"仍有 {failed} 项失败 ❌")
    sys.exit(1)
