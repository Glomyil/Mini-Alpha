"""qa/test_engine_masks.py —— T-2.3 新增行为：拆掩码 / 现金约束 / 对账

这些都是【纯夹具】测试（小面板、手算），所以进 CI ✓
原来 engine 只有 tests/test_engine.py 那种"手算钉死"的脚本，
但【现金约束的缩量分支】在真实数据上一次都没被触发过 ⇒ 这里是它唯一的覆盖。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import run_layer      # noqa: E402
from backtest.panel import derive_flags    # noqa: E402


# ═════════════════ ① derive_flags：reason → 两个方向 ═════════════════
@pytest.mark.parametrize(
    "reason, exp_buy, exp_sell, 为什么",
    [
        (0, True,  True,  "ok"),
        (1, False, False, "停牌：两个方向都封"),
        (2, False, True,  "涨停：只封买（涨停照样能卖）"),
        (3, True,  False, "跌停：只封卖（跌停照样能买）"),
        (4, False, False, "不在市：两个都封"),
        (5, False, False, "有行无价：两个都封"),
    ],
)
def test_derive_flags_mapping(reason, exp_buy, exp_sell, 为什么):
    fl = derive_flags(np.array([reason], dtype="int8"))
    assert bool(fl["can_buy"][0]) is exp_buy, f"reason={reason}（{为什么}）can_buy"
    assert bool(fl["can_sell"][0]) is exp_sell, f"reason={reason}（{为什么}）can_sell"


def test_derive_flags_涨停只封一边():
    """这条是整个 T-2.3 的语义核心：两个方向独立，不是互相取反。"""
    fl = derive_flags(np.array([2, 3], dtype="int8"))
    assert fl["can_buy"].tolist() == [False, True]
    assert fl["can_sell"].tolist() == [True, False]


# ═════════════════ 夹具 ═════════════════
def make_panel(days, codes, reason_rows, price=10.0, cash_prices=None):
    """price 恒定的小面板；reason_rows 指定每天每只票的 reason 编码。"""
    idx = pd.DatetimeIndex(days)
    px = pd.DataFrame(price, index=idx, columns=codes)
    r = pd.DataFrame(np.asarray(reason_rows, dtype="int8"), index=idx, columns=codes)
    return {
        "hfq_open": px,
        "hfq_close": px,
        "can_trade": r.isin([0, 2, 3]),      # 有数据的日子
        "reason": r,
    }


# ═════════════════ ② 涨停买不到 ⇒ 现金空着；跌停卖不掉 ⇒ 仓位留着 ═════════════════
def test_limit_up_blocks_buy_only():
    """B 涨停 ⇒ 买不进；但那天如果本来就想卖 B，就能卖。"""
    days = ["2020-01-02", "2020-01-03"]
    codes = ["A", "B"]
    panels = make_panel(days, codes, [[0, 2], [0, 2]])
    W = pd.DataFrame([[0.5, 0.5]], index=[pd.Timestamp(days[0])], columns=codes)

    res = run_layer(W, panels, [pd.Timestamp(days[0])], slippage=0.0, fee_buy=0.0, fee_sell=0.0)
    sh = res["shares"]

    assert float(sh["B"].iloc[-1]) == 0.0, "B 涨停 ⇒ 一股都买不进"
    assert float(sh["A"].iloc[-1]) > 0.0, "A 正常 ⇒ 买得进"
    assert res["cash"].iloc[-1] > 0.4, "买不到的那 50% 应该留在现金里（不补仓）"


def test_limit_down_blocks_sell_only():
    """第 2 期想把 A 卖掉，但 A 跌停 ⇒ 卖不掉，仓位被迫留着。"""
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["A", "B"]
    # d2 执行（d3 建仓 A 100%）→ d4 执行（想全换 B），但 A 已跌停
    panels = make_panel(days, codes, [[0, 4], [0, 4], [3, 0], [3, 0]])
    W = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                     index=[pd.Timestamp(days[0]), pd.Timestamp(days[2])], columns=codes)

    res = run_layer(W, panels, [pd.Timestamp(days[0]), pd.Timestamp(days[2])],
                    slippage=0.0, fee_buy=0.0, fee_sell=0.0)
    sh = res["shares"]

    assert float(sh["A"].iloc[-1]) > 0.0, "A 跌停 ⇒ 卖不掉 ⇒ 仓位必须留着"
    # scale 会缩到 ~1e-13（不是精确 0，因为 cash_avail 有一点点尾数）
    assert float(sh["B"].iloc[-1]) == pytest.approx(0.0, abs=1e-10), "A 卖不掉 ⇒ 没钱 ⇒ B 买不进"
    assert res["cash"].iloc[-1] == pytest.approx(0.0, abs=1e-12), "现金没动"


# ═════════════════ ③ 现金约束：缩量分支（真实数据上从没触发过） ═════════════════
def test_cash_constraint_scales_down_buys():
    """手里只有 50% 的钱、却想买 100% ⇒ 必须等比缩到买得起，且不透支。"""
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["A", "B"]
    # d1 执行：A 100%（现金花光）；d3 执行：想全换 B，但 A 停牌 ⇒ 卖不掉 ⇒ 只有 0 现金
    panels = make_panel(days, codes, [[0, 4], [0, 4], [1, 0], [1, 0]])
    W = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                     index=[pd.Timestamp(days[0]), pd.Timestamp(days[2])], columns=codes)

    res = run_layer(W, panels, [pd.Timestamp(days[0]), pd.Timestamp(days[2])],
                    slippage=0.0, fee_buy=0.0, fee_sell=0.0)
    sh = res["shares"]

    assert float(sh["A"].iloc[-1]) > 0.0, "A 停牌 ⇒ 卖不掉"
    assert float(sh["B"].iloc[-1]) == pytest.approx(0.0, abs=1e-12), \
        "A 卖不掉 ⇒ 没现金 ⇒ B 一股都买不进（scale≈0，不从别处补）"
    assert float(res["cash"].min()) >= -1e-9, "任何一天都不许透支"


def test_scale_keeps_cash_non_negative_with_fees():
    """带手续费时 scale 必须把费用也算进去，否则现金会透支（真踩过：-0.000641）。"""
    days = ["2020-01-02", "2020-01-03"]
    codes = ["A"]
    panels = make_panel(days, codes, [[0], [0]])
    W = pd.DataFrame([[1.0]], index=[pd.Timestamp(days[0])], columns=codes)

    res = run_layer(W, panels, [pd.Timestamp(days[0])])       # 用默认费率和滑点
    assert float(res["cash"].min()) >= -1e-9, f"现金透支到 {float(res['cash'].min()):.6e}"
    assert float(res["shares"]["A"].iloc[-1]) > 0.0, "还是买到了（只是缩了一点）"


# ═════════════════ ④ 对账：四条恒等式 ═════════════════
def test_reconcile_cash_identity():
    """① 现金恒等：cash 的变化 = 卖出 − 买入 − 费用。"""
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["A", "B"]
    panels = make_panel(days, codes, [[0, 0]] * 4)
    W = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                     index=[pd.Timestamp(days[0]), pd.Timestamp(days[2])], columns=codes)
    res = run_layer(W, panels, [pd.Timestamp(days[0]), pd.Timestamp(days[2])], slippage=0.0)

    cash = res["cash"].to_numpy()
    fees = res["fees"]
    trades = res["trades"]

    for rd in trades.index:
        # 执行日 = 调仓日的下一个交易日
        i = list(res["nav"].index).index(rd)
        prev_cash = cash[i - 1]
        # 用 trades 反推买卖额（价格恒 10）
        delta = trades.loc[rd].to_numpy()
        buy = float(np.clip(delta, 0, None).sum()) * 10.0
        sell = float(-np.clip(delta, None, 0).sum()) * 10.0
        exp = prev_cash + sell - buy - float(fees.loc[rd])
        # trades 是 float32 快照（见 engine.py 的 trades_a）⇒ 反推金额有 1e-8 级误差
        assert abs(cash[i] - exp) < 1e-6, f"{rd.date()} 现金恒等式差 {cash[i] - exp:.3e}"


def test_reconcile_nav_identity():
    """② 净值恒等（逐日）：nav = cash + Σ 股数 × 收盘价。"""
    days = ["2020-01-02", "2020-01-03", "2020-01-06"]
    codes = ["A", "B"]
    panels = make_panel(days, codes, [[0, 0]] * 3)
    W = pd.DataFrame([[0.5, 0.5]], index=[pd.Timestamp(days[0])], columns=codes)
    res = run_layer(W, panels, [pd.Timestamp(days[0])], slippage=0.0)

    nav = res["nav"].to_numpy()
    cash = res["cash"].to_numpy()
    close = panels["hfq_close"].to_numpy()
    day_pos = {d: i for i, d in enumerate(res["nav"].index)}

    # 只在【执行日】验：那天 shares 快照就是当天收盘后的真实持仓 ✓
    # （非执行日没有快照，硬套最后一个快照会在建仓前那段算错）
    for rd in res["shares"].index:
        i = day_pos[rd]
        sh = res["shares"].loc[rd].to_numpy()
        exp = cash[i] + float((sh * close[i]).sum())
        assert abs(nav[i] - exp) < 1e-6, f"{rd.date()} 净值恒等式差 {nav[i] - exp:.3e}"


def test_reconcile_trading_does_not_change_nav():
    """③ 调仓本身不改变净值 —— 只扣费用。这是最强的一条（能抓缩量/掩码/重复扣费）。"""
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["A", "B"]
    panels = make_panel(days, codes, [[0, 0]] * 4)
    W = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                     index=[pd.Timestamp(days[0]), pd.Timestamp(days[2])], columns=codes)
    res = run_layer(W, panels, [pd.Timestamp(days[0]), pd.Timestamp(days[2])],
                    slippage=0.0, fee_buy=0.0, fee_sell=0.0)      # 零成本 ⇒ 净值必须一动不动

    nav = res["nav"].to_numpy()
    # 价格全程 10、零成本 ⇒ 净值应当恒为 1.0
    assert np.allclose(nav, 1.0, atol=1e-12), f"零成本调仓后净值变了：{nav}"