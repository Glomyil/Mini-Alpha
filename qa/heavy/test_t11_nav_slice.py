"""T-1.1 的行为回归：指标输入从【首个调仓日】起算，而不是面板起点。

为什么切片：2016-01-29 之前没有持仓，那 13 个月净值恒为 1.0
⇒ 稀释波动、稀释年化、抬高夏普 ✗
修法：main.py 里 _nav = nav_frame(res).loc[rdates[0]:] 后再交给报告。

静态守门（切片位置 / meta 来源 / 目录约定）在 qa/test_static_guards.py —— 不需要数据，CI 每次都跑。
这里的两条要真面板，CI 跳过。
"""
import pathlib

import pytest

pytestmark = pytest.mark.needs_data

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_slice_drops_empty_prefix():
    """切片后第一行 == 首个调仓日，且真的丢掉了前面那段空仓期。"""
    from backtest.assemble import rebalance_dates
    from backtest.panel import load_panels

    panels = load_panels()
    rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= 2016]
    first = rdates[0]

    idx = panels["hfq_close"].index
    nav_full = idx.to_series(index=idx)          # 用索引当替身，不需要跑引擎
    nav = nav_full.loc[first:]

    assert nav.index[0] == first, f"切片起点 {nav.index[0]} != 首个调仓日 {first}"
    assert nav_full.index[0] < first, "面板起点竟然不早于首个调仓日 —— 前提不成立"
    dropped = len(nav_full) - len(nav)
    assert 100 < dropped < 400, f"切掉 {dropped} 行 —— 太少说明没生效，太多说明起点算错"


@pytest.mark.slow
def test_nav_frame_index_covers_panel():
    """引擎必须仍按全索引跑（ROADMAP 要求），否则切片将无事可做。"""
    from backtest.assemble import build_features, rebalance_dates
    from backtest.engine import nav_frame, run_all_layers
    from backtest.factors import FACTORS
    from backtest.panel import load_panels
    from backtest.weights import build_target_weights

    panels = load_panels()
    rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= 2016]
    spec = FACTORS["rev_5d"]
    fv = spec["fn"](build_features(panels, ["rev_5d"]), {}) * spec["sign"]
    W, _ = build_target_weights(fv, rdates, panels["can_trade"])
    res = run_all_layers(W, panels, rdates)
    nav_full = nav_frame(res)

    assert len(nav_full) == len(panels["hfq_close"].index), (
        f"nav_frame 长度 {len(nav_full)} != 面板 {len(panels['hfq_close'].index)} "
        "—— 引擎被改成只跑调仓日了？ROADMAP 要求引擎照旧全索引跑")
    assert nav_full.index[0] < rdates[0], "nav_frame 起点不早于首个调仓日 —— 切片将无事可做"
