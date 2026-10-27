"""qa/test_panel_reason.py —— T-2.2 的 reason 宽表

分三层：
  ① 单票层面 _reason_of()：6 个编码 + 优先级（parametrize 表）
  ② 装配层面：没有这一格 ⇒ 4
  ③ 不变量：reason 与 can_trade 的关系、编码合法、无 NaN

⚠️ 现在应该是【红】的 —— reason 还没写进 panel.py。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import panel as P          # noqa: E402

# ═══════════ ① 单票层面：6 个编码 + 优先级 ═══════════
@pytest.mark.parametrize(
    "case, susp, bad, has_px, lim_up, lim_dn, expected",
    [
        # case                     停牌  坏价  有价  涨停  跌停   期望 reason
        ("正常",                   False, False, True,  False, False, 0),
        ("停牌",                   True,  False, True,  False, False, 1),
        ("开盘涨停",               False, False, True,  True,  False, 2),
        ("开盘跌停",               False, False, True,  False, True,  3),
        ("无价（hfq_close 是 NaN）", False, False, False, False, False, 5),
        ("坏价（is_price_bad）",    False, True,  True,  False, False, 5),
        # ── 优先级：先剔除（无价）最后分类（涨跌停）──
        ("停牌 + 坏价 ⇒ 5 赢",      True,  True,  True,  False, False, 5),
        ("停牌 + 无价 ⇒ 5 赢",      True,  False, False, False, False, 5),
        ("涨停 + 坏价 ⇒ 5 赢",      False, True,  True,  True,  False, 5),
        ("停牌 + 涨停 ⇒ 2 赢（停牌先写，被涨停覆盖）", True, False, True, True, False, 2),
    ],
)
def test_reason_of(case, susp, bad, has_px, lim_up, lim_dn, expected):
    """_reason_of 收到的是【已经取反过】的布尔：no_susp / no_bad / has_px。"""
    assert hasattr(P, "_reason_of"), "panel.py 里还没有 _reason_of —— 这就是要让它变红的原因"
    r = P._reason_of(
        no_susp=np.array([not susp]),
        no_bad=np.array([not bad]),
        has_px=np.array([has_px]),
        lim_up=np.array([lim_up]),
        lim_dn=np.array([lim_dn]),
    )
    assert str(r.dtype) == "int8", f"{case}: dtype 应为 int8，实际 {r.dtype}"
    assert int(r[0]) == expected, f"{case}: 期望 {expected}，实际 {int(r[0])}"


# ═══════════ ② 装配层面：没有这一格 ⇒ 4 ═══════════
def test_missing_cell_becomes_4(tmp_path):
    """装配后，某只票没有那天 ⇒ NaN ⇒ 必须变成 4（不在市），不是 0 也不是 NaN。"""
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    a = pd.Series([0, 1, 2], index=idx, name="sh.600000", dtype="int8")
    b = pd.Series([0, 0], index=idx[:2], name="sz.000001", dtype="int8")     # 少一天
    wide = pd.concat([a, b], axis=1).sort_index().fillna(4).astype("int8")

    assert wide.loc["2020-01-06", "sz.000001"] == 4, "缺格应为 4"
    assert wide.loc["2020-01-03", "sh.600000"] == 1, "有格的值不该被改动"
    assert str(wide.dtypes.iloc[0]) == "int8"
    assert wide.notna().all().all(), "fillna(4) 之后不许留 NaN"


# ═══════════ ③ 不变量（对真面板宽表） ═══════════
CACHE = Path(__file__).resolve().parents[1] / "backtest" / "cache"


def _load_reason():
    f = sorted(CACHE.glob("*_reason.parquet"))
    assert f, "缓存里还没有 reason 宽表 —— 面板没重建，或 FIELDS 里没加 'reason'"
    return pd.read_parquet(f[0])


def test_reason_has_only_valid_codes():
    r = _load_reason()
    codes = set(np.unique(r.to_numpy().ravel()).tolist())
    assert codes <= {0, 1, 2, 3, 4, 5}, f"出现非法编码：{codes - {0,1,2,3,4,5}}"
    assert str(r.dtypes.iloc[0]) == "int8", f"dtype 应为 int8，实际 {r.dtypes.iloc[0]}"
    assert r.notna().all().all(), "reason 里还有 NaN —— fillna(4) 漏了"


def test_reason_shape_matches_can_trade():
    f = sorted(CACHE.glob("*_can_trade.parquet"))
    assert f, "缓存里没有 can_trade 宽表 —— 面板从未建过"
    ct = pd.read_parquet(f[0])
    r = _load_reason()
    assert r.shape == ct.shape, f"形状不一致：reason {r.shape} vs can_trade {ct.shape}"
    assert r.index.equals(ct.index), "行索引（日期）不一致"
    assert list(r.columns) == list(ct.columns), "列名（股票）不一致"


def test_reason_0_equals_can_trade_minus_limits():
    """reason == 0 应当等价于 can_trade 且当天没有涨跌停。"""
    f = sorted(CACHE.glob("*_can_trade.parquet"))
    if not f:
        pytest.skip("缓存里没有 can_trade 宽表")
    ct = pd.read_parquet(f[0])
    r = _load_reason()
    ok_from_reason = (r == 0)
    ok_expected = ct & ~r.isin([2, 3])       # can_trade 且不是涨跌停
    # ★ can_trade 不排除涨跌停，所以 can_trade 会包含 reason=2/3 的格子
    assert (ok_from_reason & ~ok_expected).sum().sum() == 0, "reason=0 但有不该 ok 的格子"


def test_reason_2_and_3_are_subsets_of_can_trade():
    """涨停/跌停那一天，仍然「有数据能算收益」⇒ 必须 can_trade=True。"""
    f = sorted(CACHE.glob("*_can_trade.parquet"))
    if not f:
        pytest.skip("缓存里没有 can_trade 宽表")
    ct = pd.read_parquet(f[0])
    r = _load_reason()
    for code in (2, 3):
        bad = (r == code) & ~ct
        assert int(bad.sum().sum()) == 0, f"reason=={code} 但 can_trade=False 的格子：{int(bad.sum().sum())}"