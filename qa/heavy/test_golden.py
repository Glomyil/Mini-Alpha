"""T-3.3 golden：真数据下的全期数字，改动前后必须一位不变。

为什么放 qa/heavy 而不是 qa/：
  它要 load_panels() + run_all_layers()，秒级起步，而且依赖本地 1.2 GB 数据。
  qa/ 的约定是「纯假数据、进 CI」；heavy 层是「真数据、手动跑」。

这条测试的价值：
  以后任何人改 report.py / engine.py / metrics.py / panel.py 导致全期结果漂移，
  它立刻红 —— 而 qa/test_segment.py 那 7 条用的是假数据，抓不到这种漂移。

⚠️ 数据一更新（下次增量拉新交易日），这些数会变 ——
   那时要重新钉，并在下面写清【何时因为什么】改的。别直接把断言放宽。

2026-11-02 钉：
  区间 2016-01-29 ~ 2026-10-29，130 期，面板 (2871, 5476)
  这组数来自 2026-11-01 那次增量更新（+32 个交易日 + 补 63 只清单遗漏的票）
  历史验证：切回 2026-09-07 重算仍是 -6.3944% / 5.1765%，与台账 T-2.4 一致
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CACHE = Path(__file__).resolve().parents[2] / "backtest" / "cache"

# 和其它 heavy 文件统一用 needs_data（CI 上由工作流跳过）；
# 额外加一条 skipif：本机也可能还没跑过 main.py，那时面板缓存不存在
pytestmark = [
    pytest.mark.needs_data,
    pytest.mark.skipif(
        not (CACHE / "panel_clean_full_2015_2026_meta.json").exists(),
        reason="面板缓存不存在（没跑过 main.py）—— heavy 测试跳过",
    ),
]


def _nav_for(factor):
    from backtest.panel import load_panels
    from backtest.assemble import build_features, rebalance_dates
    from backtest.factors import FACTORS
    from backtest.weights import build_target_weights
    from backtest.engine import run_all_layers, nav_frame

    panels = load_panels()
    # rule 必传；用 "ME" —— 下面的 golden 数字（130 期 / 13.44%）就是月频口径
    rdates = [d for d in rebalance_dates(panels["hfq_close"].index, rule="ME")
              if d.year >= 2016]
    spec = FACTORS[factor]
    fval = spec["fn"](build_features(panels, [factor]), {}) * spec["sign"]
    W, _ = build_target_weights(fval, rdates, panels["can_trade"])
    res = run_all_layers(W, panels, rdates)
    return nav_frame(res).loc[rdates[0]:], panels, rdates


def test_amihud_layer4_annualized_is_pinned():
    """amihud_20d 全期层4 年化 = 13.44%（±0.05pp）。

    这是 T-3.3 改动（split 进 spec + 两个新报告函数）的回归锚点：
    改动只是【加输出】，不该动到任何一个已有数字。
    """
    from backtest.metrics import ann_return

    nav, panels, rdates = _nav_for("amihud_20d")

    # 先钉住形状：形状变了说明数据换了，下面的年化也就不可比了
    #
    # 2026-11-09 数据更新：股票清单从 5476 → 5477（多一只新上市），
    #   实测影响：层4 年化 13.44% → 13.46%、超额 5.98% → 6.00%（都 +0.02pp），
    #   基准年化 7.46% 不变、调仓期数 130 不变。
    #   ⇒ 按本测试的要求"重新钉数"而不是放宽容差：下面两个数已钉到新实测值。
    assert panels["hfq_close"].shape == (2871, 5477), (
        f"面板形状变了（{panels['hfq_close'].shape}）—— 数据更新过？"
        "那时要重新钉下面每一个数，而不是放宽断言")
    assert len(rdates) == 130, f"调仓期数变了（{len(rdates)}），原为 130"

    got = float(ann_return(nav[4]))
    assert got == pytest.approx(0.1346, abs=5e-4), (
        f"层4 年化 {got:.4%} != 13.44% —— 改动动到了主路径，或数据换了")


def test_amihud_layer4_excess_is_pinned():
    """amihud_20d 全期层4 超额 = +5.98%（±0.05pp）。

    超额依赖【基准】，所以这条同时守着 metrics.equal_weight_nav（T-3.1）。
    """
    from backtest.metrics import ann_return, equal_weight_nav

    nav, panels, rdates = _nav_for("amihud_20d")
    bench = equal_weight_nav(panels).loc[rdates[0]:]

    excess = float(ann_return(nav[4]) - ann_return(bench))
    assert excess == pytest.approx(0.0600, abs=5e-4), (
        f"层4 超额 {excess:.4%} != +5.98%")