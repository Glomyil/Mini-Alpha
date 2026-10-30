"""T-3.1 全 A 等权基准 —— 两个功能，一个文件。

  metrics.equal_weight_nav  —— 算基准净值
  report.build_report       —— 第 1 节加「基准年化」「超额」两列

为什么不放进 qa/test_unit_smoke.py：那个文件的定位是「纯函数 vs pandas 对拍」，
而基准没有 pandas 的对拍对象（pandas 里没有「全 A 等权」这种东西）。
这里测的是【业务语义】：停牌票算不算 0 收益、全天不可交易怎么办、
没数据时报告写不写「缺基准数据」。

全部手造夹具，不依赖本地数据，秒级。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import metrics as MT          # noqa: E402
from backtest import report as RP           # noqa: E402


# ── 最小夹具：2 天 / 2 只票 ─────────────────────────────
IDX = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])


def _panel(ret, can_trade):
    """只造 equal_weight_nav 要的两张表。"""
    return {
        "ret_1d_hfq": pd.DataFrame(ret, index=IDX, columns=["A", "B"], dtype="float64"),
        "can_trade": pd.DataFrame(can_trade, index=IDX, columns=["A", "B"], dtype=bool),
    }


# ── ① 两票涨跌相抵 => 净值恰好 1.0 ─────────────────────
def test_offsetting_returns_give_flat_nav():
    """A +10%、B -10% 且都可交易 => 等权收益 0% => 净值停在 1.0。

    这条守的是「别算成平均价格」：价格平均会让 A 的权重被放大 10 倍（A 是 100 元、B 是 10 元），
    算出来是 +8.6% 而不是 0%。
    """
    panels = _panel([[np.nan, np.nan], [0.10, -0.10]], [[True, True], [True, True]])
    nav = MT.equal_weight_nav(panels)

    assert float(nav.iloc[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(nav.iloc[1]) == pytest.approx(1.0, abs=1e-12)


# ── ② 一只停牌 => 只算剩下那只 ────────────────────────
def test_suspended_name_is_excluded_not_treated_as_zero():
    """B 停牌（can_trade=False），A +10% => 基准收益 10%，不是 5%。

    停牌日的 ret_1d_hfq 是 0（前收重复）。如果不过滤，等于给停牌票记「0 收益」，
    基准会被系统性拉平 —— 面板里 25% 的格子都是停牌，这个偏差会很大。
    """
    panels = _panel([[np.nan, np.nan], [0.10, 0.0]], [[True, True], [True, False]])
    nav = MT.equal_weight_nav(panels)

    assert float(nav.iloc[1]) == pytest.approx(1.10, abs=1e-12), (
        f"停牌票被当成 0 收益了：净值 {float(nav.iloc[1]):.6f}（应为 1.10，若是 1.05 就说明没过滤）")


# ── ③ 全天一只都不可交易 => 那天收益 0，不能断 ─────────
def test_all_untradable_day_yields_zero_not_nan():
    """整行 can_trade 全 False => mean 是 NaN => 必须 fillna(0.0) 承接。

    不承接的话净值从那一天起全变 NaN，后面所有年化都是 nan。
    """
    panels = _panel([[np.nan, np.nan], [0.5, 0.5]], [[True, True], [False, False]])
    nav = MT.equal_weight_nav(panels)

    assert not nav.isna().any(), f"净值里有 NaN：{nav.tolist()}"
    assert float(nav.iloc[-1]) == pytest.approx(1.0, abs=1e-12)


# ── ④ 连涨两天 => 累乘不是加总 ────────────────────────
def test_returns_compound_not_sum():
    """两天各 +10% => 1.1 × 1.1 = 1.21，不是 1.20。"""
    idx3 = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    panels = {
        "ret_1d_hfq": pd.DataFrame([[np.nan, np.nan], [0.10, 0.10], [0.10, 0.10]],
                                   index=idx3, columns=["A", "B"], dtype="float64"),
        "can_trade": pd.DataFrame(True, index=idx3, columns=["A", "B"], dtype=bool),
    }
    nav = MT.equal_weight_nav(panels)

    assert float(nav.iloc[-1]) == pytest.approx(1.21, abs=1e-12)


# ── ⑤ 报告里真的出现那两列 ───────────────────────────
def _report_fixture(with_panels):
    """凑一套 build_report 能跑起来的最小输入。"""
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    nav_df = pd.DataFrame({L: np.linspace(1.0, 1.0 + 0.02 * L, 40) for L in range(5)}, index=idx)
    rdates = [idx[0], idx[10], idx[20], idx[30], idx[-1]]

    panels = None
    if with_panels:
        panels = {
            "hfq_close": pd.DataFrame(np.full((40, 1), 10.0), index=idx, columns=[0]),
            "can_trade": pd.DataFrame(True, index=idx, columns=[0]),
            "ret_1d_hfq": pd.DataFrame(np.full((40, 1), 0.001), index=idx, columns=[0]),
        }
    # factor_val 的列数必须等于 panels["hfq_close"] 的列数（rank_ic 有断言）
    fval = pd.DataFrame(np.random.RandomState(0).randn(40, 1), index=idx, columns=[0])
    return nav_df, panels, rdates, fval


def test_report_contains_benchmark_columns(tmp_path):
    """panels 齐全 => 第 1 节必须有「基准年化」和「超额」两列。"""
    nav_df, panels, rdates, fval = _report_fixture(with_panels=True)
    out = RP.build_report("rev_20d", nav_df, panels, rdates, None,
                          outdir=tmp_path, factor_val=fval)
    text = out["md"].read_text(encoding="utf-8")

    assert "基准年化" in text, "报告第 1 节缺「基准年化」列"
    assert "超额" in text, "报告第 1 节缺「超额」列"


# ── ⑥ 没数据 => 明写「缺基准数据」，不许悄悄跳过 ───────
def test_report_says_missing_when_no_panels(tmp_path):
    """panels=None => 报告写「缺基准数据」（ROADMAP 明确要求，不许静默跳过）。"""
    nav_df, _, rdates, fval = _report_fixture(with_panels=False)
    out = RP.build_report("rev_20d", nav_df, None, rdates, None,
                          outdir=tmp_path, factor_val=fval)
    text = out["md"].read_text(encoding="utf-8")

    assert "缺基准数据" in text, "没有基准数据时报告应当明说，而不是整列消失"
