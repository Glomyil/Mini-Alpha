"""qa/test_limits.py —— 涨跌停派生（T-2.1）的规格

分七块：
  ① 板块判定        ② 涨跌幅比率      ③ 新股期天数
  ④ 舍入            ⑤ 10 行 fixture 真值表（parametrize）
  ⑥ run() 集成      ⑦ 未知板块 raise

纯函数直接喂 numpy 数组 —— 不造 DataFrame、不跑 pipeline，所以能进 CI。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cleaner.limits import (          # noqa: E402
    _round_cent, board_of, compute_limit_pct, compute_limits,
    compute_new_listing_days, run,
)


# ─────────────────────── ① 板块判定 ───────────────────────
def test_board_mapping():
    assert board_of("sh.600000") == "main"
    assert board_of("sz.000001") == "main"
    assert board_of("sz.300750") == "chinext"
    assert board_of("sz.301001") == "chinext"
    assert board_of("sz.302132") == "chinext"   # 302 是创业板新增号段（数据反推确认 20%）
    assert board_of("sh.688981") == "star"
    assert board_of("sh.689009") == "star"
    assert board_of("bj.830799") == "bse"
    assert board_of("bj.430047") == "bse"


# ─────────────────── ② 涨跌幅比率 ───────────────────
def test_pct_main_st_vs_normal():
    """ST 只对主板生效：ST ⇒ 5%，普通 ⇒ 10%。"""
    pct = compute_limit_pct("main", [pd.Timestamp("2020-01-02")] * 3, [0, 1, 0])
    assert list(pct) == [0.10, 0.05, 0.10]


def test_pct_chinext_before_and_after_2020_08_24():
    """创业板改革：2020-08-24 当天起就是 20%。"""
    dates = [pd.Timestamp("2020-08-21"), pd.Timestamp("2020-08-24")]
    pct = compute_limit_pct("chinext", dates, [0, 0])
    assert pct[0] == pytest.approx(0.10)
    assert pct[1] == pytest.approx(0.20)


def test_pct_star_and_bse():
    """科创板 20%，北交所 30%（本库无数据，只能断言参数）。"""
    star = compute_limit_pct("star", [pd.Timestamp("2021-01-04")] * 2, [0, 1])
    bse = compute_limit_pct("bse", [pd.Timestamp("2021-01-04")] * 2, [0, 0])
    assert list(star) == [0.20, 0.20]        # ST 不覆盖科创板
    assert list(bse) == [0.30, 0.30]


# ─────────────────── ③ 新股期天数 ───────────────────
def test_new_listing_days_main_before_and_after_2023_04_10():
    """主板：全面注册制前只有上市首日，之后是前 5 日。"""
    before = compute_new_listing_days("main", [pd.Timestamp("2023-04-07")])
    after = compute_new_listing_days("main", [pd.Timestamp("2023-04-10")])
    assert int(before[0]) == 1
    assert int(after[0]) == 5


def test_new_listing_days_other_boards_always_5():
    for board in ("chinext", "star", "bse"):
        d = compute_new_listing_days(board, [pd.Timestamp("2016-01-04")])
        assert int(d[0]) == 5, board


# ─────────────────── ④ 舍入 ───────────────────
def test_round_cent_is_not_python_round():
    """10.35×1.1 = 11.385 —— Python round() 给 11.38（银行家舍入）✗，交易所要 11.39 ✓。"""
    assert float(_round_cent(11.385)) == 11.39
    assert float(_round_cent(0.125)) == 0.13      # round(0.125,2)=0.12 ✗
    assert float(_round_cent(2.675)) == 2.68      # round(2.675,2)=2.67 ✗


# ─────────── ⑤ 10 行 fixture 真值表（每行断言 3 个字段）───────────
# 说明：「退市」无法用单行表达（它是"之后没有行"），由数据形态体现，不是 flag 能测的；
#      「未知板块」需要 pytest.raises，放在第 ⑦ 块。
@pytest.mark.parametrize(
    "case, prev, open_px, pct, days, n_days, changed, tradeable,"
    " exp_side, exp_limit, exp_flag_up, exp_flag_dn, resumed",
    [
        # case                    prev    open    pct   days n_days chg    trd    side  limit  up dn  resumed
        ("一字板涨停",             10.00, 11.00, 0.10,  99,   5, False, True,  "up", 11.00, 1, 0, False),
        ("开盘涨停盘中打开",        10.00, 11.00, 0.10,  99,   5, False, True,  "up", 11.00, 1, 0, False),
        ("跌停开盘",               10.00,  9.00, 0.10,  99,   5, False, True,  "dn",  9.00, 0, 1, False),
        ("ST 主板恰好 -5%",        10.00,  9.50, 0.05,  99,   5, False, True,  "dn",  9.50, 0, 1, False),
        ("停牌日不判",             10.00, 11.00, 0.10,  99,   5, False, False, "up", 11.00, 0, 0, False),
        ("复权因子≠1 不影响限价",   10.00, 11.00, 0.10,  99,   5, False, True,  "up", 11.00, 1, 0, False),
        ("完全正常",               10.00, 10.20, 0.10,  99,   5, False, True,  "up", 11.00, 0, 0, False),
        ("主板上市首日涨 43%",     10.00, 14.30, 0.10,   1,   1, False, True,  "up", 11.00, 0, 0, False),
        ("创业板第 3 日涨 25%",    10.00, 12.50, 0.20,   3,   5, False, True,  "up", 12.00, 0, 0, False),
        ("★ 除权日跌 10%",         12.00, 10.80, 0.10,  99,   5, True,  True,  "dn", 10.80, 0, 0, False),
        ("★ 复牌首日（无涨跌幅）",  10.00,  9.00, 0.10,  99,   5, False, True,  "dn",  9.00, 0, 0, True),
    ],
)
def test_fixture_table(case, prev, open_px, pct, days, n_days, changed, tradeable,
                       exp_side, exp_limit, exp_flag_up, exp_flag_dn, resumed):
    r = compute_limits(
        prev_close=np.array([prev]),
        open_px=np.array([open_px]),
        pct=np.array([pct]),
        days_since_listing=np.array([days], dtype="int16"),
        new_listing_days=np.array([n_days], dtype="int16"),
        factor_changed=np.array([changed]),
        tradeable=np.array([tradeable]),
        resumed=np.array([resumed]),
    )
    limit = r["limit_up"][0] if exp_side == "up" else r["limit_down"][0]
    assert float(limit) == pytest.approx(exp_limit), f"{case}: 限价不对"
    assert int(r["flag_up"][0]) == exp_flag_up, f"{case}: flag_up 不对"
    assert int(r["flag_down"][0]) == exp_flag_dn, f"{case}: flag_down 不对"


# ─────────────────── ⑥ run() 集成 ───────────────────
def _fake_df(n=40):
    """造一只主板票：第 10 天涨停、第 11 天起价格变了、第 20 天除权。"""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    d = pd.DataFrame({
        "date": dates, "code": "sh.600000",
        "open": [10.00] * n, "close": [10.00] * n,
        "high": [10.00] * n, "low": [10.00] * n,
        "volume": [1000] * n, "amount": [1e5] * n,
        "back_adj_factor": [1.0] * n, "isST": [0] * n, "tradestatus": [1] * n,
    })
    d.loc[10, ["open", "close", "high", "low"]] = 11.00    # 涨停（prev_close = 10.00）
    d.loc[11, "open"] = 11.50
    for i in range(20, n):                                  # 除权之后因子保持 1.10
        d.loc[i, "back_adj_factor"] = 1.10
    d.loc[20, "open"] = 9.99                                # 除权日跳空
    return d


def test_run_adds_three_flag_columns_as_int8():
    out = run(_fake_df(), {}, {"code": "sh.600000", "asset_type": "stock"})
    for col in ("flag_open_limit_up", "flag_open_limit_down", "flag_new_listing"):
        assert col in out.columns, col
        assert str(out[col].dtype) == "int8", (col, out[col].dtype)


def test_run_attrs_has_six_counters():
    out = run(_fake_df(), {}, {"code": "sh.600000", "asset_type": "stock"})
    attrs = out.attrs["limits"]
    assert set(attrs) == {"n_limit_up", "n_limit_down", "n_new_listing",
                          "n_factor_changed", "n_missing_prev", "n_resumed"}
    assert attrs["n_missing_prev"] == 1        # 只有首行没有前收
    assert attrs["n_factor_changed"] == 1      # 第 20 天变一次
    assert attrs["n_limit_up"] >= 1            # 第 10 天那个涨停


# ─────────────────── ⑦ 未知板块 ───────────────────
def test_unknown_board_raises():
    with pytest.raises(ValueError):
        board_of("sh.999888")