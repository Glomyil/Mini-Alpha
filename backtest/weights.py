from __future__ import annotations
import pandas as pd
import numpy as np
from backtest.assemble import build_features, rebalance_dates
from backtest.factors import FACTORS


def build_target_weights(factor, rdates, can_trade, n_layers=5):
    cols = factor.columns
    # —— 健康度断言（2026-10-13）：把"数据退化"变成响亮失败，而不是静默算错 ——
    _ct = can_trade.to_numpy()
    assert _ct.size and _ct.any(), f"can_trade 全 False（{can_trade.shape}）—— 数据退化，拒绝计算权重"
    if _ct.shape[1] > 1000:                            # 全市场面板才做区间校验（合成小面板会越界）
        _r = float(_ct.mean())
        assert 0.4 < _r < 0.95, f"can_trade True 占比 {_r:.3f} 越界（健康值 ≈0.73）—— 数据可疑"
    layers = pd.DataFrame(-1, index=rdates, columns=cols, dtype="int8")
    W = [pd.DataFrame(0.0, index=rdates, columns=cols, dtype="float32")
         for _ in range(n_layers)]
    for dt in rdates:
        if dt not in factor.index or dt not in can_trade.index:
            continue
        v = factor.loc[dt]
        ok = can_trade.loc[dt].fillna(False) & v.notna()
        tickers = v.index[ok]
        if len(tickers) == 0:
            continue
        n = len(tickers)
        r = v[tickers].rank(method="first")
        layer = ((r.to_numpy()-1)*n_layers//n).astype("int8")
        layers.loc[dt, tickers] = layer

        for L in range(n_layers):
            members = tickers[layer == L]
            if len(members):
                W[L].loc[dt, members] = np.float32(1.0 / len(members))
    return W, layers


def weights_for_factor(name, panels, rdates, n_layers=5):
    """便利函数（冒烟/研究用）：因子名 → (W, layers)

    rdates 必须由调用方给：研究区间（例如"2016 起"）是**研究口径**，不住在这个函数里 ——
    留默认值会让"不传就全区间"和"传 2016 起"变成两条静默分叉的路（141 vs 129 行，且不报错）。
    """
    spec = FACTORS[name]
    feats = build_features(panels, [name], {})
    val = spec["fn"](feats, {}) * spec["sign"]          # 定向：方向只乘这一次
    return build_target_weights(val, rdates, panels["can_trade"], n_layers)


def main():
    import time
    from backtest.panel import load_panels

    t0 = time.time()
    panels = load_panels()
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index) if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    print(
        f"面板 {panels['hfq_close'].shape} ｜ 调仓日 {len(rdates)} 个 ｜ 因子 rev_20d（已定向）｜ 分 {len(W)} 层 ｜ 用时 {time.time() - t0:.1f}s")
    print()
    print(f"{'层':<4}{'平均只数':>10}{'权重和 均值':>14}{'权重和 min':>12}{'权重和 max':>12}")
    for L in range(len(W)):
        cnt = (layers == L).sum(axis=1)
        s = W[L].sum(axis=1)
        print(
            f"{L:<4}{cnt.mean():>10.0f}{s.mean():>14.4f}{s.min():>12.4f}{s.max():>12.4f}")
    print()
    print(f"未入层格子占比：{(layers == -1).to_numpy().mean():.2%}（不可交易 / 无因子值）")
    print(f"空层行数（那天一只可交易都没有）：{int((W[0].sum(axis=1) == 0).sum())}")
    print("下一步（engine 那一格）：W[层] → T+1 开盘撮合 → 净值 / 换手 / 费用")


if __name__ == "__main__":
    main()
