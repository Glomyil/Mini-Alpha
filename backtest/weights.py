from __future__ import annotations
import os
import pandas as pd
import numpy as np
from backtest.assemble import build_features, rebalance_dates
from backtest.factors import FACTORS
from backtest.metrics import rank1d
from loguru import logger


def _assign_layers(F, C, n_layers):
    """核心循环：名次 -> 层 + 层内等权。抽出来是为了能【算两遍自证】。

    所有输出数组都用 np.full / np.zeros 预填，结构上不存在"未写区"；
    r 用 np.empty 但 r[order] 里 order 是排列、写满每一个格子。
    """
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
    return LAY, Wn


def build_target_weights(factor, rdates, can_trade, n_layers=5, *, select=None):
    """按因子值分 n_layers 层，返回 [(权重表, ...), 分层表]。

    select：可选的【选股池】（点位2：invest_pool ∩ can_trade）。
        None（默认）⇒ 行为与加这个参数之前一字不差。
        收窄它只会让"可选范围"变小，**不会**改变可交易性。
        例：沪深300 只有 300 只 ⇒ select 后池子占比 ~5%，
        但 can_trade 仍是全市场的 73% ⇒ 下面的健康断言不能被它带偏。
    """
    cols = factor.columns
    # --- health asserts (2026-10-13): data degradation must fail loudly ---
    # ⚠️ 断言只看 can_trade（市场事实）。若改成看 select，把宇宙缩到 hs300 就会
    #    误报 "can_trade True ratio 0.055 out of range"（实测踩过）——
    #    那不是数据退化，是我们自己收窄了选股池。哨兵要留，尺子不能换。
    _ct = can_trade.to_numpy()
    assert _ct.size and _ct.any(
    ), f"can_trade all False ({can_trade.shape}) -- data degraded, refusing to compute weights"
    if _ct.shape[1] > 1000:
        _r = float(_ct.mean())
        assert 0.4 < _r < 0.95, f"can_trade True ratio {_r:.3f} out of range (healthy ~0.73) -- data suspect"

    # 选股池：can_trade ∩ select（select=None 时就是 can_trade）
    _pool = can_trade if select is None else (can_trade & select)

    # --- pure numpy core: no pandas row-level .loc inside the loop (PITFALLS #23) ---
    # 每张表各 reindex 一次，然后拍成 C 连续的快照 —— 这之后全是纯 numpy，不再回到 pandas
    F = np.ascontiguousarray(
        factor.reindex(index=rdates, columns=cols).to_numpy(dtype="float64", copy=True))
    C = np.ascontiguousarray(
        _pool.reindex(index=rdates, columns=cols).fillna(False).to_numpy(dtype=bool, copy=True))

    LAY, Wn = _assign_layers(F, C, n_layers)

    # 自证（2026-11-18）：同样的输入再算一遍，结果必须逐字节相同。
    # 这是对 PITFALLS #23/#30「偶发读到未写入页」的结构化解法 ——
    # 不需要知道 bug 在哪一层，只要它不稳定，这里当场炸。
    # 关掉：set QD_SELFCHECK=0（跑重活时省一次循环，约 +0.3s）
    if os.environ.get("QD_SELFCHECK", "1") != "0":
        LAY2, Wn2 = _assign_layers(F, C, n_layers)
        if not np.array_equal(LAY, LAY2):
            _d = int((LAY != LAY2).sum())
            raise AssertionError(
                f"层表两次计算不一致：{_d} 格不同（共 {LAY.size}）"
                f" —— 读到未写入内存，PITFALLS #23")
        for _L, (_a, _b) in enumerate(zip(Wn, Wn2)):
            if not np.array_equal(_a, _b):
                raise AssertionError(
                    f"第 {_L} 层权重两次计算不一致 —— 同上（PITFALLS #23）")

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
        panels["hfq_close"].index, rule="ME") if d.year >= 2016]
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