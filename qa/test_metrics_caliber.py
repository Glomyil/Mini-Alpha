"""T-6.3 口径改造：IC 标签与多空期收益自 T+1 开盘起算。

三条测试都是「在旧实现上必红」的：
  · 标签如果回退成 close->close，截面排序会反过来，IC 从 +1 变成 -1
  · 多空期收益如果回退成收盘净值，期数会多一期且数值不同
  · build_report 如果静默回落收盘口径，最后一条会挂
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import metrics as MT


def _two_stock_panel():
    """6 个交易日、2 只票。故意让 open 与 close 的组间排序【相反】。

    open : a 恒为 10，b 每期翻倍  -> 标签 a < b
    close: a 每期 +50%，b 恒为 10 -> 标签 a > b
    """
    idx = pd.bdate_range("2020-01-01", periods=6)
    open_ = pd.DataFrame({"a": [100, 10, 100, 10, 100, 10],
                         "b": [100, 10, 100, 20, 100, 40]}, index=idx)
    close = pd.DataFrame({"a": [10, 10, 20, 20, 30, 30],
                         "b": [10, 10, 10, 10, 10, 10]}, index=idx)
    ok = pd.DataFrame(True, index=idx, columns=["a", "b"])
    return idx, open_, close, ok


def test_rank_ic_label_uses_open_of_exec_date_not_close_of_signal_date():
    """标签必须是 open(成交日) -> open(下一成交日)，不是 close(信号日) -> close。"""
    idx, open_, close, ok = _two_stock_panel()
    rdates = [idx[0], idx[2], idx[4]]          # 成交日 = idx[1], idx[3], idx[5]
    # 因子值：a 恒小于 b（排序 a < b）
    factor = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]}, index=rdates)

    ic = MT.rank_ic(factor, {"hfq_close": close, "hfq_open": open_, "can_trade": ok},
                    rdates, min_names=2)

    assert len(ic) == 2, f"应有 2 期，实际 {len(ic)}"
    # open 标签：两期都是 a(0%) < b(+100%)，与因子同序 => IC = +1
    # 若回退成 close 标签：a(+100%/+50%) > b(0%)，与因子反序 => IC = -1
    assert np.allclose(ic.to_numpy(), 1.0), (
        f"IC 应为 +1（开盘口径），实际 {ic.to_numpy()} —— 是不是回退成收盘标签了？")


def test_rank_ic_requires_hfq_open():
    """缺 hfq_open 必须当场 KeyError，不许静默回落到收盘（DEBT-011）。"""
    idx, _, close, ok = _two_stock_panel()
    factor = pd.DataFrame({0: [1.0, 1.0]}, index=[idx[0], idx[2]])
    with pytest.raises(KeyError):
        MT.rank_ic(factor, {"hfq_close": close, "can_trade": ok}, [idx[0], idx[2]])


def test_ls_summary_period_count_survives_trailing_exec_day():
    """最后一个调仓日的成交日是它的【下一个交易日】—— 期数不能因此少一期。

    这条钉的是「开盘净值必须传未切片的那份」：切成 until=最后调仓日 就会丢一期。
    """
    idx = pd.bdate_range("2020-01-01", periods=10)
    cols = [0, 1]
    # 收盘净值只到 idx[7]（最后一期调仓日），开盘净值却多一天
    nav_close = pd.DataFrame({c: np.linspace(1.0, 1.1, 8) for c in cols}, index=idx[:8])
    nav_open = pd.DataFrame({c: np.linspace(1.0, 1.2, 10) for c in cols}, index=idx)
    rdates = [idx[0], idx[3], idx[6], idx[7]]

    ls = MT.ls_summary(nav_close, nav_open, rdates, periods_per_year=12)

    assert ls["期数"] == 3, (
        f"4 个调仓日应得 3 期，实际 {ls['期数']} —— "
        f"开盘净值被切成 until=最后调仓日 了？")


def test_ls_summary_rejects_missing_nav_open():
    """nav_open_df 是必需位置参数：漏传必须 TypeError，不许有默认值。"""
    idx = pd.bdate_range("2020-01-01", periods=6)
    nav = pd.DataFrame({0: np.linspace(1, 1.1, 6), 1: np.linspace(1, 1.2, 6)}, index=idx)
    with pytest.raises(TypeError):
        MT.ls_summary(nav, [idx[0], idx[2], idx[4]])    # noqa: 故意少传一个参数

def test_build_report_refuses_to_silently_fall_back_to_close(tmp_path):
    """没有开盘净值时必须 ValueError —— 这是 DEBT-011 的硬约束，不许静默回落。

    用 tmp_path fixture 而不是 tempfile：沙箱建不了系统 TEMP 子目录
    （qa/conftest.py 已把 tmp_path 挪到 logs/.pytest_tmp/）。
    """
    from backtest import report as RP
    idx = pd.bdate_range("2020-01-01", periods=6)
    nav = pd.DataFrame({0: np.linspace(1.0, 0.9, 6), 1: np.linspace(1.0, 1.1, 6)}, index=idx)
    rdates = [idx[0], idx[2], idx[4]]
    with pytest.raises(ValueError, match="开盘净值"):
        RP.build_report("x", nav, None, rdates, None,   # res=None => 无 nav_open
                        outdir=tmp_path)


def test_rank_ic_skips_period_when_no_next_exec_day():
    """最后一个调仓日没有成交日（b+1 越界）=> 该期跳过，并且不崩。"""
    idx = pd.bdate_range("2020-01-01", periods=3)
    px = pd.DataFrame({"a": [10.0, 11.0, 12.0], "b": [10.0, 10.5, 11.0]}, index=idx)
    ok = pd.DataFrame(True, index=idx, columns=["a", "b"])
    # 3 个调仓日 = 全部日期；末期 b=2 的成交日 3 越界
    factor = pd.DataFrame({"a": [2.0, 2.0], "b": [1.0, 1.0]}, index=idx[:2])
    ic = MT.rank_ic(factor, {"hfq_close": px, "hfq_open": px, "can_trade": ok},
                    list(idx), min_names=2)
    assert len(ic) == 1, f"越界那期必须跳过，应剩 1 期，实际 {len(ic)}"