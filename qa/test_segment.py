"""T-3.3 分段分析 —— 三个公开接口，一个文件。

  report.build_report(..., split=)   —— 报告能不能接上分段（T1 T2）
  report.segment_table               —— 分段算法本身（T3 T4 T5）
  report.yearly_ic_table             —— 年度 IC 分组（T6）

为什么单独一个文件而不是并进 test_benchmark.py：
  那个文件的定位是「基准线」（T-3.1），这里的是「分段」（T-3.3）。
  按被测模块分文件是 qa/ 的约定，不按 T 票号。

全部手造夹具，不依赖本地数据，秒级。
真数据下的 golden 值在 qa/heavy/test_golden.py（不进 CI）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import report as RP          # noqa: E402


def _report_fixture(with_panels=False):
    """凑一套 build_report 能跑起来的最小输入。

    注意：index 是 40 个工作日（2020-01 起），所以 split 必须落在这 40 天之内 ——
    传一个区间外的日期会触发 segment_table 的 ValueError（那是 T5 在守的）。
    """
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    nav_df = pd.DataFrame({L: np.linspace(1.0, 1.0 + 0.02 * L, 40) for L in range(5)},
                          index=idx)
    rdates = [idx[0], idx[10], idx[20], idx[30], idx[-1]]
    panels = None
    if with_panels:
        panels = {
            "hfq_close": pd.DataFrame(np.full((40, 1), 10.0), index=idx, columns=[0]),
            "can_trade": pd.DataFrame(True, index=idx, columns=[0]),
            "ret_1d_hfq": pd.DataFrame(np.full((40, 1), 0.001), index=idx, columns=[0]),
        }
    fval = pd.DataFrame(np.random.RandomState(0).randn(40, 1), index=idx, columns=[0])
    return nav_df, panels, rdates, fval, idx


# ── T1 split 为空 => 报告里【没有】分段节 ────────────────
def test_empty_split_produces_no_segment_section(tmp_path):
    """split="" => 不分段，报告不出现 ## 2.，out["segment"] 是 None。

    这条守的是「向后兼容」：老调用不传 split 时行为和 T-3.3 之前完全一致，
    报告不多出一节。
    """
    nav_df, panels, rdates, fval, _ = _report_fixture()
    out = RP.build_report("rev_20d", nav_df, panels, rdates, None,
                          outdir=tmp_path, factor_val=fval, split="")
    text = out["md"].read_text(encoding="utf-8")

    assert "## 2. 分段对比" not in text, "split 为空却出了分段节"
    assert out["segment"] is None, f"split 为空时 segment 应为 None，实际 {type(out['segment'])}"


# ── T2 split 有值 => 报告里有分段节（2026-11-02 TypeError 的回归）──
def test_nonempty_split_produces_segment_section(tmp_path):
    """split 落在区间内 => 报告出现 ## 2. 分段对比，且 segment 是两行表。

    ★ 这条是 2026-11-02 那次崩溃的回归：
      main.py 传了 split=spec.split，但 build_report 的签名没加这个参数，
      于是 python main.py 在第一个因子就 TypeError。
      这条测试在旧代码上会红（unexpected keyword argument），现在是绿的。

    为什么不顺带断言 ## 3. 年度 IC：
      panels=None 时算不出 IC，§3 本来就会跳过。§3 的守由 T6 单独负责。
    """
    nav_df, panels, rdates, fval, idx = _report_fixture()
    cut = idx[20].strftime("%Y-%m-%d")          # fixture 区间内，不是区间外
    out = RP.build_report("rev_20d", nav_df, panels, rdates, None,
                          outdir=tmp_path, factor_val=fval, split=cut)
    text = out["md"].read_text(encoding="utf-8")

    assert "## 2. 分段对比" in text, f"split={cut} 却没出分段节"
    assert out["segment"] is not None, "segment 表没返回"
    assert len(out["segment"]) == 2, f"单分界应切 2 段，实际 {len(out['segment'])}"

def _month_end_nav(seq, m=5):
    """把一条【明确的数值序列】包成 5 层 nav，index 是 12 个月末。

    为什么用显式序列而不是"前 half 涨、后 half 平"的生成器：
      生成器里"第 i 期的增长率"和"第 i 期属于哪一段"是两套下标，
      一不小心就错一位（本文件第一版就错了两次）。
      直接写出每个时点的净值，预期值就不用推了。
    """
    assert len(seq) == 12, f"需要 12 个月末，实际 {len(seq)}"
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    col = pd.Series(seq, index=idx, dtype="float64")
    return pd.DataFrame({L: col * (1.0 + 0.01 * L) for L in range(m)}, index=idx)


# 12 个月末的基准序列：前 5 期从 1.0 涨到 2.0（翻倍），后 6 期停在 2.0
# split 放在 idx[5]（2020-06-30）时：
#   段1 = idx[0..5] => 1.0 → 2.0   => 翻倍
#   段2 = idx[5..11] => 2.0 → 2.0  => 原地不动
_FLAT_AFTER_DOUBLE = [1.0] + [2.0 ** (k / 5.0) for k in range(1, 6)] + [2.0] * 6


def _flat_nav():
    """全平的 12 期 nav（每期 1.0），供边界/越界测试用。"""
    return _month_end_nav([1.0 + 0.01 * i for i in range(12)])


# ── T3 split 落在两期之间 => 吸附到【前一期】────────────
def test_split_between_two_periods_snaps_to_previous():
    """split=2020-08-15（不是月末）=> 段1 的「止」应是 2020-07-31。

    这条守的是 searchsorted(side="right") - 1 这个吸附规则。
    如果实现改成 side="left" 或者向上取整，段1 会变成 2020-08-31，
    于是「8 月上半月」被算进了后段 —— 边界悄悄挪了一个月。
    """
    nav = _flat_nav()
    seg = RP.segment_table(nav, "2020-08-15")

    assert len(seg) == 2, f"单分界应 2 段，实际 {len(seg)}"
    assert str(seg.iloc[0]["止"]) == "2020-07-31", (
        f"段1 的止应为 2020-07-31（split 之前最后一个交易日），实际 {seg.iloc[0]['止']}")
    assert str(seg.iloc[1]["起"]) == "2020-07-31", "段2 的起应接段1 的止（同一天，nav.loc[a:b] 两头都含）"


# ── T4 每段【独立】算年化，不是拿全期的数 ────────────────
def test_each_segment_annualizes_independently():
    """前 6 期翻倍、后 6 期原地不动 => 段1 年化巨大，段2 年化 ≈ 0。

    这条守的是「段内独立 ann_return(seg)」。
    如果实现错拿全期 nav 算年化再除以段数，两段的数会一样 —— 那时这条会红。

    注意：归一化（norm = s / s.iloc[0]）对 ann_return 没有影响（都是比值），
    所以这条测不出归一化 —— 归一化只在【返回期末净值】时才有意义。
    """
    nav = _month_end_nav(_FLAT_AFTER_DOUBLE)
    seg = RP.segment_table(nav, "2020-06-30")
    a1 = float(seg.iloc[0]["层4年化"])
    a2 = float(seg.iloc[1]["层4年化"])

    assert a1 > 1.0, f"段1（翻倍）年化应 >100%，实际 {a1:.4f}"
    assert abs(a2) < 1e-9, f"段2（原地不动）年化应 ≈0，实际 {a2:.10f}"
    assert abs(a1 - a2) > 0.5, "两段年化几乎相同 => 很可能用了全期的数，而不是各段独立算"


# ── T5 split 越界 => 必须报错，不许静默跳过 ─────────────
def test_split_outside_range_raises():
    """split 早于首个交易日 => ValueError，不静默产空表。

    静默跳过的话，用户以为分段算了，实际报告里那一节是空的或错的（PITFALLS 1096）。
    """
    nav = _flat_nav()
    with pytest.raises(ValueError, match="不在回测区间"):
        RP.segment_table(nav, "1999-01-01")

# ── T6 年度 IC 分组：期数 / 均值 / 胜率 ─────────────────
def test_yearly_ic_groups_by_calendar_year():
    """2016 年 12 个值（7 正 5 负）、2017 年 12 个（3 正 9 负）=>
    期数 [12, 12]、IC胜率 [7/12, 3/12]。

    预期值是【手数的】：12 个数里 7 个正，胜率就是 7/12 —— 不拿实现重算一遍，
    否则这条测试永远绿（tautological）。

    为什么不测 IC 均值对不对：那是 rank_ic 的活，这里只测【分组和胜率】这两个
    yearly_ic_table 自己的行为。
    """
    idx16 = pd.date_range("2016-01-31", periods=12, freq="ME")
    idx17 = pd.date_range("2017-01-31", periods=12, freq="ME")
    # 2016：前 7 个正、后 5 个负；2017：前 3 个正、后 9 个负
    vals = [0.10] * 7 + [-0.05] * 5 + [0.08] * 3 + [-0.06] * 9
    ic = pd.Series(vals, index=idx16.append(idx17), dtype="float64")

    tab = RP.yearly_ic_table(ic)

    assert list(tab.index) == [2016, 2017], f"年份分组错：{list(tab.index)}"
    assert list(tab["期数"]) == [12, 12], f"期数错：{list(tab['期数'])}"
    assert float(tab.loc[2016, "IC胜率"]) == pytest.approx(7 / 12, abs=1e-12), (
        f"2016 胜率应 7/12，实际 {float(tab.loc[2016, 'IC胜率']):.6f}")
    assert float(tab.loc[2017, "IC胜率"]) == pytest.approx(3 / 12, abs=1e-12), (
        f"2017 胜率应 3/12，实际 {float(tab.loc[2017, 'IC胜率']):.6f}")


# ── T7 年度表遇到 NaN 要丢掉，且期数跟着变 ──────────────
def test_yearly_ic_drops_nan_and_counts_shrink():
    """IC 里有 NaN（某期算不出来）=> dropna 后该年期数变小，不许把 NaN 当 0 计。

    把 NaN 当 0 会让胜率被稀释（0 不算正也不算负，但会进分母）。
    """
    idx = pd.date_range("2016-01-31", periods=4, freq="ME")
    ic = pd.Series([0.10, np.nan, -0.05, np.nan], index=idx, dtype="float64")

    tab = RP.yearly_ic_table(ic)

    assert int(tab.loc[2016, "期数"]) == 2, f"NaN 应被丢掉，期数应为 2，实际 {int(tab.loc[2016, '期数'])}"
    assert float(tab.loc[2016, "IC胜率"]) == pytest.approx(0.5, abs=1e-12), "1 正 1 负 => 胜率 0.5"