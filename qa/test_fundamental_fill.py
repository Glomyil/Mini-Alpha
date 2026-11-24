"""fundamental_fill 的行为测试（2026-11-23）。

seam：`run(df, params, ctx, kind)` —— 唯一对外接口，测试只碰这里。

重点锁住「季报和年报混合」的四条规则：
  ① 同日公告多期 ⇒ 取新公告，旧期整行作废（旧期独有字段变 NaN）
  ② 某期字段缺失 ⇒ 该窗口内保持 NaN，不向上借（禁 ffill）
  ③ 首份公告之前 ⇒ 全 NaN（禁 bfill/前视）
  ④ 五张表各用各自的 pubDate_<表> ⇒ 同一行可来自不同期

期望值全部手算/来自 spec，不由实现推导。夹具是 39 列最小事件表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cleaner import fundamental_fill as FF
from data.fundamental_merge import TABLES

_ALL_IND = [c for cols in TABLES.values() for c in cols]


def _event(rows):
    """rows: [{statDate, pubDate_profit, roeAvg, MBRevenue, ...}] → 39 列事件表。"""
    base = {c: np.nan for c in _ALL_IND}
    for t in TABLES:
        base[f"pubDate_{t}"] = pd.NaT
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    cols = ["statDate"] + [f"pubDate_{t}" for t in TABLES] + _ALL_IND
    df = pd.DataFrame(out, columns=cols)
    df["statDate"] = pd.to_datetime(df["statDate"])
    for t in TABLES:
        df[f"pubDate_{t}"] = pd.to_datetime(df[f"pubDate_{t}"])
    return df.sort_values("statDate").reset_index(drop=True)


def _days(*ds):
    return pd.DatetimeIndex(pd.to_datetime(list(ds)))


# ── ① 同日公告多期：取新公告 ────────────────────────────
def test_same_day_annual_and_q1_merges_by_column():
    """年报与一季报同一天公告 ⇒ 该日披露的信息取【并集】，逐列独立。

    规则（PITFALLS #34 补充）：
      同一 pubDate 的多期构成该日的一次完整披露，
      每列取「当天所有报告里最后一次出现的非空值」。

    手算（两组 pubDate 都是 2020-04-25，按 statDate 升序后 groupby().last()）：
      roeAvg     —— 两期都有 ⇒ 取更晚的 2020Q1 ⇒ 0.05
      MBRevenue  —— 只有年报有 ⇒ 取年报的那条 ⇒ 1.0e10（一季报是 NaN 不参与）
    """
    df = _event([
        {"statDate": "2019-12-31", "pubDate_profit": "2020-04-25",
         "roeAvg": 0.20, "MBRevenue": 1.0e10},
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25",
         "roeAvg": 0.05, "MBRevenue": np.nan},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2020-04-24", "2020-04-27")})

    # 公告前：一期都没公告过 ⇒ 全 NaN（不因合并而提前可见）
    assert np.isnan(out.loc[pd.Timestamp("2020-04-24"), "roeAvg"])
    assert np.isnan(out.loc[pd.Timestamp("2020-04-24"), "MBRevenue"])

    row = out.loc[pd.Timestamp("2020-04-27")]
    assert row["roeAvg"] == pytest.approx(0.05), "两期都有值时取更晚的那一期"
    assert row["MBRevenue"] == pytest.approx(1.0e10), \
        "只有年报披露的字段必须可见 —— 那天市场确实看到了它"


def test_missing_field_does_not_borrow_from_previous_period():
    """某期该字段为 NaN ⇒ 该窗口内一直是 NaN，不继承上一期（禁 ffill）。"""
    df = _event([
        {"statDate": "2019-12-31", "pubDate_profit": "2020-01-10",
         "MBRevenue": 2.0e10},
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25",
         "MBRevenue": np.nan},
    ])
    out = FF.run(df, {}, {"trading_days": _days(
        "2020-02-03", "2020-04-27", "2020-06-01")})

    assert out.loc[pd.Timestamp("2020-02-03"), "MBRevenue"] == pytest.approx(2.0e10)
    assert np.isnan(out.loc[pd.Timestamp("2020-04-27"), "MBRevenue"])
    assert np.isnan(out.loc[pd.Timestamp("2020-06-01"), "MBRevenue"])


# ── ③ 首公告之前 ────────────────────────────────────────
def test_nothing_before_first_announcement():
    """首份公告之前必须全 NaN —— 往回填就是前视。"""
    df = _event([
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25", "roeAvg": 0.05},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2020-01-02", "2020-04-24", "2020-04-27")})
    assert out.loc[pd.Timestamp("2020-01-02"), _ALL_IND].isna().all()
    assert out.loc[pd.Timestamp("2020-04-24"), _ALL_IND].isna().all()
    assert out.loc[pd.Timestamp("2020-04-27"), "roeAvg"] == pytest.approx(0.05)


# ── ④ 五张表各用各自的 pubDate ─────────────────────────
def test_each_table_uses_its_own_pubdate():
    """同一行里 balance 与 profit 可来自不同期 —— 这是 #33 的核心。

    手算：2020-02-03 那天
      balance 已公告 2020-01-10 那期（2019Q4）⇒ currentRatio=0.9，lag=24 天
      profit  尚未公告（pubDate 2020-04-10）⇒ 用 2019-10-31 那期，lag=95 天
    """
    df = _event([
        {"statDate": "2019-09-30", "pubDate_profit": "2019-10-31",
         "roeAvg": 0.30, "pubDate_balance": "2019-10-31", "currentRatio": 0.8},
        {"statDate": "2019-12-31", "pubDate_profit": "2020-04-10",
         "roeAvg": 0.40, "pubDate_balance": "2020-01-10", "currentRatio": 0.9},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2020-02-03")})
    row = out.loc[pd.Timestamp("2020-02-03")]

    assert row["currentRatio"] == pytest.approx(0.9)   # balance 用 2019Q4
    assert row["roeAvg"] == pytest.approx(0.30)        # profit 仍用 2019Q3
    assert int(row["lag_balance"]) == 24               # 2020-02-03 − 2020-01-10
    assert int(row["lag_profit"]) == 95                # 2020-02-03 − 2019-10-31


def test_lag_counts_from_the_published_date():
    """lag = 交易日 − 该表可见的 pubDate，按天算，同日公告当天可见。"""
    df = _event([
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25", "roeAvg": 0.05},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2020-04-25", "2020-05-05")})
    assert int(out.loc[pd.Timestamp("2020-04-25"), "lag_profit"]) == 0   # 当天可见
    assert int(out.loc[pd.Timestamp("2020-05-05"), "lag_profit"]) == 10


# ── 公告日倒序（已评估不改，锁住现状）─────────────────
def test_later_period_published_earlier_still_takes_latest_announcement():
    """后一期比前一期【更早】公告时（实测 0.156%），取最近公告的那份。

    这是已评估并接受的现行为：语义是「最近公告」而不是「最新期次」。
    锁住它，免得将来有人无意中改了还没人发现。
    """
    df = _event([
        {"statDate": "2020-09-30", "pubDate_profit": "2021-09-01", "roeAvg": 0.11},
        {"statDate": "2020-12-31", "pubDate_profit": "2021-02-01", "roeAvg": 0.22},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2021-03-01", "2021-10-01")})
    # 2021-03-01：只有 2021-02-01 那份可见 ⇒ 用 2020Q4（0.22）
    assert out.loc[pd.Timestamp("2021-03-01"), "roeAvg"] == pytest.approx(0.22)
    # 2021-10-01：2021-09-01 那份也公告了，且它 pubDate 更晚 ⇒ 取它 = 2020Q3
    # 这就是「最近公告」语义：账期反而变旧了。已评估接受，此处锁住。
    assert out.loc[pd.Timestamp("2021-10-01"), "roeAvg"] == pytest.approx(0.11)


# ── 形状与可查性 ────────────────────────────────────────
def test_output_shape_columns_and_index():
    """行数 = 交易日数，43 列（32 指标 + 5 lag + 5 pubDate + statDate），索引升序。"""
    df = _event([
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25", "roeAvg": 0.05},
    ])
    days = _days("2020-04-27", "2020-01-02")          # 故意乱序传入
    out = FF.run(df, {}, {"trading_days": days})
    assert len(out) == 2
    assert len(out.columns) == 43
    assert sorted(out.columns) == sorted(
        _ALL_IND + [f'lag_{t}' for t in TABLES] + [f'pubDate_{t}' for t in TABLES] + ['statDate'])
    assert out.index.is_monotonic_increasing, "索引必须升序，不随传入顺序"


def test_entirely_missing_table_is_reported_in_attrs():
    """整张表空（没下到）必须能从 attrs 查出来 —— #34 第 3 类。"""
    df = _event([
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25", "roeAvg": 0.05},
    ])
    out = FF.run(df, {}, {"trading_days": _days("2020-04-27")})
    a = out.attrs["fundamental_fill"]
    assert a["tables_entirely_missing"] == [
        "balance", "cashflow", "growth", "operation"]   # sorted()，字母序
    assert a["periods_without_pubdate"]["balance"] == 1
    assert a["source_periods"] == 1
    assert a["rows_with_no_fundamental"] == 0


# ── 入口守卫 ────────────────────────────────────────────
def test_empty_event_table_raises():
    with pytest.raises(ValueError):
        FF.run(_event([]), {}, {"trading_days": _days("2020-04-27")})


def test_missing_trading_days_raises():
    """ctx 缺 trading_days ⇒ 当场 KeyError，不等到环节里 KeyError。

    注意守卫顺序：check_not_empty 在前，所以夹具必须非空，
    否则会先撞 ValueError（空表），测不到 trading_days 这一层。
    """
    df = _event([{"statDate": "2020-03-31", "pubDate_profit": "2020-04-25"}])
    with pytest.raises(KeyError, match="trading_days"):
        FF.run(df, {}, {})