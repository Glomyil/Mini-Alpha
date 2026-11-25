"""merge_daily 的行为测试。

seam：`merge(price_df, fund_df)` —— 唯一对外接口。

期望值来自教学文档 §6 的验收清单，不由实现推导。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cleaner import fundamental_fill as FF
from data.cleaner import merge_daily as MD


def _price(*days):
    """行情表：最低限度的合法形状（date 升序）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(list(days)))
    return pd.DataFrame({"date": idx, "hfq_close": 1.0, "hfq_open": 1.0})


def _fund(days, **values):
    """财务表：43 列齐全、index 是 date、**没有 date 列**。"""
    cols = FF.output_columns()
    dt_cols = [c for c in cols if c == "statDate" or c.startswith("pubDate_")]
    idx = pd.DatetimeIndex(pd.to_datetime(list(days)))
    df = pd.DataFrame(index=idx)
    for c in dt_cols:
        df[c] = pd.NaT
    for c in cols:
        if c not in dt_cols:
            df[c] = np.nan
    for c, v in values.items():
        df[c] = v
    return df


# ── 分支 B：fund = None（93% 的票走这条）────────────────
def test_none_branch_fills_all_fund_columns():
    """缺财务数据时补全 43 列，数值列 float64、日期列 datetime64[ns]。"""
    out = MD.merge(_price("2020-01-02", "2020-01-03"), None)
    cols = FF.output_columns()

    assert len(out) == 2
    assert list(out.columns) == list(_price("2020-01-02", "2020-01-03").columns) + cols
    for c in cols:
        assert out[c].isna().all(), c
    assert out["statDate"].dtype == "datetime64[ns]"
    assert out["pubDate_profit"].dtype == "datetime64[ns]"
    assert out["roeAvg"].dtype == "float64"


def test_none_branch_and_data_branch_have_identical_columns():
    """两条分支的列名与顺序必须完全一致 —— 否则 panel 拼宽表会静默出错。"""
    px = _price("2020-01-02", "2020-01-03")
    days = ["2020-01-02", "2020-01-03"]
    a = MD.merge(px, None)
    b = MD.merge(px, _fund(days, roeAvg=0.1))
    assert list(a.columns) == list(b.columns)
    assert list(a.dtypes) == list(b.dtypes)


# ── 分支 A：有财务数据 ──────────────────────────────────
def test_price_columns_come_first_and_are_untouched():
    """前 N 列必须是行情原样，财务列追加在后。"""
    px = _price("2020-01-02", "2020-01-03")
    out = MD.merge(px, _fund(["2020-01-02", "2020-01-03"], roeAvg=[0.1, 0.2]))

    assert list(out.columns[: len(px.columns)]) == list(px.columns)
    for c in px.columns:
        pd.testing.assert_series_equal(out[c], px[c].rename(c))


def test_values_come_from_the_matching_date():
    """按 date 一一对齐，不是按位置。"""
    px = _price("2020-01-02", "2020-01-03")
    out = MD.merge(px, _fund(["2020-01-02", "2020-01-03"], roeAvg=[0.1, 0.2]))
    assert out["roeAvg"].tolist() == pytest.approx([0.1, 0.2])


# ── 分支 C：fund 区间比 price 短 ────────────────────────
def test_short_fund_leaves_nan_but_keeps_rows():
    """财务表短于行情表 ⇒ 多出的行是 NaN，行数绝不变（不缩样本）。"""
    px = _price("2020-01-02", "2020-01-03", "2020-01-06")
    fund = _fund(["2020-01-03"], roeAvg=[0.9])
    out = MD.merge(px, fund)

    assert len(out) == 3
    assert np.isnan(out["roeAvg"].iloc[0])
    assert out["roeAvg"].iloc[1] == pytest.approx(0.9)
    assert np.isnan(out["roeAvg"].iloc[2])


# ── 分支 D/E：两种失败必须响亮 ─────────────────────────
def test_duplicate_fund_index_raises():
    """fund 索引重复 ⇒ validate="one_to_one" 抛错，不静默去重。"""
    px = _price("2020-01-02")
    fund = _fund(["2020-01-02", "2020-01-02"], roeAvg=[0.1, 0.2])
    with pytest.raises(Exception):
        MD.merge(px, fund)


def test_unsorted_price_raises():
    """price 未升序 ⇒ 入口 assert 抛错，不静默 sort。"""
    px = _price("2020-01-03", "2020-01-02")
    with pytest.raises(AssertionError, match="升序"):
        MD.merge(px, None)


def test_missing_date_column_raises():
    """price 缺 date 列 ⇒ check_required_columns 报错。"""
    with pytest.raises(ValueError):
        MD.merge(pd.DataFrame({"hfq_close": [1.0]}), None)


# ── attrs：统计报告 ────────────────────────────────────
def test_attrs_counts_rows_with_fundamental():
    """rows_with_fundamental = 至少一列非空的行数。"""
    px = _price("2020-01-02", "2020-01-03", "2020-01-06")
    fund = _fund(["2020-01-02", "2020-01-03", "2020-01-06"],
                 roeAvg=[np.nan, 0.2, np.nan])
    out = MD.merge(px, fund)
    assert out.attrs["merge_daily"]["rows_with_fundamental"] == 1


def test_attrs_lists_all_nan_columns():
    """fund_cols_all_nan 列出这票上真正全空的列。"""
    px = _price("2020-01-02")
    fund = _fund(["2020-01-02"], roeAvg=0.1)
    out = MD.merge(px, fund)
    allnan = out.attrs["merge_daily"]["fund_cols_all_nan"]
    assert "roeAvg" not in allnan
    assert "npMargin" in allnan
    assert len(allnan) == 42


def test_attrs_none_branch_reports_zero_rows_with_fundamental():
    """fund=None ⇒ rows_with_fundamental 必须是 0，全 42 列都空。"""
    out = MD.merge(_price("2020-01-02", "2020-01-03"), None)
    a = out.attrs["merge_daily"]
    assert a["fund_present"] is False
    assert a["rows_with_fundamental"] == 0
    assert len(a["fund_cols_all_nan"]) == 43      # 42 个值列 + statDate