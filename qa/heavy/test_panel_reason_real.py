"""qa/heavy/test_panel_reason_real.py —— reason 宽表的真实数据不变量（CI 跳过）

这 4 条要读 backtest/cache/*_reason.parquet —— CI 上没有缓存（也没有 cleaner_daily），
所以必须放 heavy 这一侧。原来误放在 qa/ 里，CI 直接红了 ✓
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.needs_data

CACHE = ROOT / "backtest" / "cache"


def _load(kind):
    f = sorted(CACHE.glob(f"*_{kind}.parquet"))
    assert f, f"缓存里没有 {kind} 宽表 —— 面板没重建过，或 FIELDS 里没加"
    return pd.read_parquet(f[0])


def _load_reason():
    return _load("reason")


# ═══════════ ③ 不变量（对真面板宽表） ═══════════
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