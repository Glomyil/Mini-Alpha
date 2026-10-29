from __future__ import annotations
import pandas as pd
import numpy as np
from backtest.assemble import build_features, rebalance_dates
from backtest.factors import FACTORS
from backtest.metrics import rank1d
from loguru import logger


def build_target_weights(factor, rdates, can_trade, n_layers=5):
    cols = factor.columns
    # --- health asserts (2026-10-13): data degradation must fail loudly ---
    _ct = can_trade.to_numpy()
    assert _ct.size and _ct.any(
    ), f"can_trade all False ({can_trade.shape}) -- data degraded, refusing to compute weights"
    if _ct.shape[1] > 1000:
        _r = float(_ct.mean())
        assert 0.4 < _r < 0.95, f"can_trade True ratio {_r:.3f} out of range (healthy ~0.73) -- data suspect"

    # --- pure numpy core: no pandas row-level .loc inside the loop (PITFALLS #23) ---
    # 每张表各 reindex 一次，然后拍成 C 连续的快照 —— 这之后全是纯 numpy，不再回到 pandas
    F = np.ascontiguousarray(
        factor.reindex(index=rdates, columns=cols).to_numpy(dtype="float64", copy=True))
    C = np.ascontiguousarray(
        can_trade.reindex(index=rdates, columns=cols).fillna(False).to_numpy(dtype=bool, copy=True))

    nT, nN = F.shape
    LAY = np.full((nT, nN), -1, dtype="int8")
    Wn = [np.zeros((nT, nN), dtype="float32") for _ in range(n_layers)]
    for i in range(nT):
        v = F[i]
        idx = np.flatnonzero(C[i] & np.isfinite(v))
        n = idx.size
        if n == 0:
            continue
        order = np.argsort(v[idx], kind="stable")
        r = np.empty(n, dtype="float64")
        r[order] = np.arange(1, n + 1, dtype="float64")
        layer = ((r - 1) * n_layers // n).astype("int8")
        LAY[i, idx] = layer
        for L in range(n_layers):
            sel = idx[layer == L]
            if sel.size:
                Wn[L][i, sel] = np.float32(1.0 / sel.size)

    ix = pd.Index(rdates)
    layers = pd.DataFrame(LAY, index=ix, columns=cols)
    W = [pd.DataFrame(w, index=ix, columns=cols) for w in Wn]
    for _L in range(len(W)):
        W[_L].attrs["np"] = Wn[_L]        # hand the raw ndarray to the engine: zero extra copy

    _un = int((LAY == -1).sum())
    _wmin = min(float(w.sum(axis=1).min()) for w in Wn)
    _empty = sum(1 for w in Wn if float(w.sum()) == 0.0)
    logger.info(f"wts unlayered:{_un} empty_layers:{int(_empty)} w_rowsum_min:{_wmin:.4f}")
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