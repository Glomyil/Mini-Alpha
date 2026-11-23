"""fund_quality 的行为测试（2026-11-23）。

核心是【单位修复】：上游在 2024-06~2025-09 把 liabilityToAsset 除以了 100。
校验断言只覆盖两条：灾难探测 + 最小样本门槛（校验精度在 ops/_fund_health.py）。

夹具是手写小表，期望值来自手算，不从实现反推：
  真值 liab=0.5 ⇒ assetToEquity=1/(1-0.5)=2.0，implied=0.5
  /100 错的存法 = 0.5/100 = 0.005 ⇒ ratio = 0.005/0.5 = 0.01
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cleaner import fund_quality as FQ


def _df(pairs):
    """pairs: [(assetToEquity, liabilityToAsset), ...] → 最小可用事件表。"""
    n = len(pairs)
    return pd.DataFrame({
        "statDate": pd.to_datetime([f"20{18 + i}-12-31" for i in range(n)]),
        "pubDate_max": pd.to_datetime([f"20{19 + i}-03-31" for i in range(n)]),
        "assetToEquity": [p[0] for p in pairs],
        "liabilityToAsset": [p[1] for p in pairs],
    })


# ── 修复本身（核心）───────────────────────────────────────
def test_fixes_exact_hundredth():
    """ratio 精确等于 0.01 的行：×100 回到真值。"""
    out = FQ.run(_df([(2.0, 0.005)]), {}, {})
    assert out.loc[0, "liabilityToAsset"] == pytest.approx(0.5)


def test_normal_rows_untouched():
    """ratio 等于 1 的行：一个字节都不许动。"""
    out = FQ.run(_df([(2.0, 0.5)]), {}, {})
    assert out.loc[0, "liabilityToAsset"] == pytest.approx(0.5)
    assert out.loc[0, "fund_fix_bits"] == 0


def test_mixed_rows_are_judged_individually():
    """同一只票里有的 /100、有的正常（2025-12-31 那种过渡期）⇒ 逐行判定。

    不能按时间段一刀切，也不能因为一行坏就整列乘 100。
    """
    out = FQ.run(_df([(2.0, 0.005), (2.0, 0.5), (2.0, 0.005)]), {}, {})
    vals = out["liabilityToAsset"].tolist()
    assert vals == pytest.approx([0.5, 0.5, 0.5])
    assert out["fund_fix_bits"].tolist() == [1, 0, 1]


def test_identity_holds_after_repair():
    """这条是修完之后唯一该信的自证：恒等式必须成立。"""
    # 每行的 /100 错法：真值 = 1-1/a2e，存成 真值/100
    #   a2e=2.0 → 真 0.5000 → 存 0.005000
    #   a2e=4.0 → 真 0.7500 → 存 0.007500
    #   a2e=1.5 → 真 0.3333 → 存 0.003333
    #   a2e=3.0 → 真 0.6667 → 存 0.006667
    df = _df([(2.0, 0.005), (4.0, 0.0075), (1.5, 0.00333333), (3.0, 0.00666667)])
    out = FQ.run(df, {}, {})
    implied = 1.0 - 1.0 / out["assetToEquity"]
    resid = (out["liabilityToAsset"] - implied).abs()
    assert float(resid.max()) < 0.01, f"修复后仍有残差 {resid.tolist()}"


def test_near_miss_is_not_repaired():
    """ratio = 0.005（差 200 倍）不是已知的 /100 ⇒ 不许猜，不许动。"""
    out = FQ.run(_df([(2.0, 0.0025)]), {}, {})
    assert out.loc[0, "liabilityToAsset"] == pytest.approx(0.0025)
    assert out.loc[0, "fund_fix_bits"] == 0


def test_nan_rows_untouched_and_not_counted():
    """liab 缺失 ⇒ 无法判定，不修、不计数。"""
    out = FQ.run(_df([(2.0, np.nan), (2.0, 0.005)]), {}, {})
    assert np.isnan(out.loc[0, "liabilityToAsset"])
    assert out.attrs["fund_quality"]["checked"] == 1
    assert out.attrs["fund_quality"]["fixed"] == 1


# ── 输出形状与记账 ────────────────────────────────────────
def test_shape_preserved_and_flag_column_added():
    """行数不变；只多一列 fund_fix_bits。"""
    df = _df([(2.0, 0.005), (2.0, 0.5)])
    out = FQ.run(df, {}, {})
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns) + ["fund_fix_bits"]
    assert out["fund_fix_bits"].dtype == np.int64


def test_attrs_recorded():
    """attrs 要能回答：查了几行、修了几行、修复率多少。"""
    out = FQ.run(_df([(2.0, 0.005), (2.0, 0.5), (2.0, 0.005)]), {}, {})
    a = out.attrs["fund_quality"]
    assert a["rows"] == 3
    assert a["checked"] == 3
    assert a["fixed"] == 2
    assert a["fix_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ── 断言：灾难探测 + 最小样本门槛 ─────────────────────────
def test_short_history_does_not_assert():
    """n < 20 时即使 100% 需要修也不炸 —— 小样本比例没有统计意义。"""
    df = _df([(2.0, 0.005)] * 5)
    out = FQ.run(df, {}, {})          # 不抛即通过
    assert out.attrs["fund_quality"]["fix_rate"] == pytest.approx(1.0)


def test_catastrophic_fix_rate_asserts():
    """n >= 20 且 100% 需要修 ⇒ 上游整体坏了，必须当场炸。"""
    df = _df([(2.0, 0.005)] * 25)
    with pytest.raises(AssertionError, match="单位异常率"):
        FQ.run(df, {}, {})


def test_unknown_breakage_asserts():
    """ratio 既不是 1 也不是 1/100 ⇒ 「未知坏法」，必须炸（不是猜着修）。"""
    df = _df([(2.0, 0.0025)] * 25)
    with pytest.raises(AssertionError, match="未知方式破口"):
        FQ.run(df, {}, {})


# ── 入口守卫 ──────────────────────────────────────────────
def test_empty_df_raises():
    """空表当场报错，不让坏盘子往下流。"""
    with pytest.raises(ValueError):
        FQ.run(pd.DataFrame({"assetToEquity": [], "liabilityToAsset": []}), {}, {})


def test_missing_required_column_raises():
    """缺 assetToEquity ⇒ 契约破裂，报错而不是静默跳过。"""
    df = pd.DataFrame({"liabilityToAsset": [0.5]})
    with pytest.raises(ValueError, match="缺了这些必须列"):
        FQ.run(df, {}, {})