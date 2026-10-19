from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger


_PANEL_SNAP = {}


def _panel_arrays(panels, cols):
    """Convert + validate the three big panel tables ONCE per process (was once per factor).

    Verification matters: an incomplete snapshot made nav_pre too small, which showed up as
    turnover ~5 instead of ~100 (PITFALLS #24). Caching also removes 17 of 18 copies.
    """
    key = (id(panels["hfq_close"]), id(panels["hfq_open"]), id(panels["can_trade"]), len(cols))
    if _PANEL_SNAP.get("key") == key:
        return _PANEL_SNAP["op"], _PANEL_SNAP["vl"], _PANEL_SNAP["ct"]

    op = np.ascontiguousarray(panels["hfq_open"].to_numpy())
    _cl = np.ascontiguousarray(panels["hfq_close"].to_numpy())
    _idx = np.where(~np.isnan(_cl), np.arange(_cl.shape[0])[:, None], 0)
    np.maximum.accumulate(_idx, axis=0, out=_idx)
    vl = np.take_along_axis(_cl, _idx, axis=0)          # numpy ffill (== pandas .ffill())
    ct = np.ascontiguousarray(panels["can_trade"].to_numpy())

    assert op.shape == vl.shape == ct.shape, "panel arrays have inconsistent shapes"
    _m = float(ct.mean())
    assert 0.4 < _m < 0.95, f"can_trade True ratio {_m:.3f} out of range -- snapshot degraded"
    assert float(np.isfinite(op).mean()) > 0.3, "hfq_open mostly non-finite -- snapshot degraded"
    assert float(np.isfinite(vl).mean()) > 0.3, "hfq_close ffill mostly non-finite -- snapshot degraded"

    _PANEL_SNAP.update(key=key, op=op, vl=vl, ct=ct)
    return op, vl, ct


def run_layer(W_layer, panels, rdates, *, fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005, init_cash=1.0):
    days = panels["hfq_close"].index
    cols = W_layer.columns
    # 入口全部走 numpy（2026-10-13 改）：不再用 pandas 的 `[cols]` / `.eq(True)` / `.copy()` 链
    # —— 那条链被实测到会**偶发**产出"形状正确、dtype 正确、但数据全 False"的表（非确定性缺陷）
    # dtype 跟随面板（真实面板 float32 / 合成面板 float64）
    # No pandas [cols]/ffill here: that chain is the one observed to occasionally return
    # shape-correct but value-wrong tables (PITFALLS #23/#24 -> nav_pre too small -> turnover ~5)
    op, vl, ct = _panel_arrays(panels, cols)
    # open valuation price: today's open if present, else previous close (pv = vl shifted by 1)
    pv = np.empty_like(vl)
    pv[0] = np.nan
    pv[1:] = vl[:-1]
    opv = np.where(np.isnan(op), pv, op)   # cached + validated (PITFALLS #24)

    # —— 健康度断言（2026-10-13 加 / 2026-10-14 优化）：把"数据退化"变成响亮失败，而不是静默算错 ——
    # 廉价的那条放这里；"扫整表"的三条挪到 exec_days 之后，只查引擎真正用到的那几十行（快 ~20 倍）
    assert ct.size and ct.any(), f"can_trade 全 False（{ct.shape}）—— 数据退化，拒绝回测"

    pos = days.get_indexer(pd.DatetimeIndex(rdates))
    exec_pos = {}
    for p, dt in zip(pos, pd.DatetimeIndex(rdates)):
        if p >= 0 and p + 1 < len(days):
            exec_pos[p+1] = dt
    exec_days = [days[i]for i in sorted(exec_pos)]

    # —— 执行日健康度（2026-10-14）：只查"真正会被成交用到"的那 ~129 行 ——
    # 原先扫整表（2839 行 × 2 张表）每次调用多花 ~120 ms，而 run_layer 一次运行要调 30 次 → 白花 2~4 s
    if ct.shape[1] > 1000 and exec_days:
        _rows = days.get_indexer(pd.DatetimeIndex(exec_days))
        _r = float(ct[_rows].mean())
        assert 0.4 < _r < 0.95, f"执行日 can_trade True 占比 {_r:.3f} 越界（健康值 ≈0.73）—— 数据可疑"
        _op = float((op[_rows] > 0.0).mean())
        _vl = float((vl[_rows] > 0.0).mean())
        assert _op > 0.3, f"执行日 hfq_open 正价占比 {_op:.3f} 过低（健康值 ≈0.75）—— 价格表退化，拒绝回测"
        assert _vl > 0.3, f"执行日 hfq_close 正价占比 {_vl:.3f} 过低（健康值 ≈0.75）—— 价格表退化，拒绝回测"

    # ---- state: all numpy, no pandas inside the loop (PITFALLS #24) ----
    n_days = len(days)
    n_cols = len(cols)
    n_ex = len(exec_days)
    shares = np.zeros(n_cols, dtype="float64")
    cash = float(init_cash)
    nav_a = np.full(n_days, np.nan, dtype="float64")
    nav_open_a = np.full(n_days, np.nan, dtype="float64")
    cash_a = np.full(n_days, np.nan, dtype="float64")
    nav_pre_a = np.full(n_ex, np.nan, dtype="float64")
    turnover_a = np.zeros(n_ex, dtype="float64")
    fees_a = np.zeros(n_ex, dtype="float64")
    trades_a = np.zeros((n_ex, n_cols), dtype="float32")
    shares_a = np.zeros((n_ex, n_cols), dtype="float32")
    n_empty = 0

    # one-shot snapshots so the loop never touches pandas
    # Prefer the ndarray handed over by build_target_weights: zero copy, so there is no
    # window in which an incomplete copy could zero/repeat rows (symptom: turnover < 20).
    Wn = getattr(W_layer, "attrs", {}).get("np")
    if Wn is not None:
        Wn = np.ascontiguousarray(Wn)
    else:
        Wn = np.nan_to_num(W_layer.to_numpy(dtype="float64", copy=True), nan=0.0)
    _rs = np.abs(Wn).sum(axis=1)
    assert float(_rs.min()) > 0.9, (
        f"W has a bad row (min row sum {float(_rs.min()):.4f}) -- refusing to run the engine")
    rd_row = {d: i for i, d in enumerate(W_layer.index)}
    ex_row = {d: k for k, d in enumerate(exec_days)}

    for i, dt in enumerate(days):
        if i in exec_pos:
            k = ex_row[dt]
            w = Wn[rd_row[exec_pos[i]]]
            tradable = ct[i] & (op[i] > 0.0)
            if not tradable.any():
                n_empty += 1
            exec_px = np.where(tradable, op[i], np.nan)
            val_px = np.nan_to_num(opv[i])
            nav_pre = cash + float((shares * val_px).sum())
            nav_pre_a[k] = nav_pre
            tgt = np.where(np.isnan(exec_px), shares,
                           nav_pre * w / np.nan_to_num(exec_px, nan=1.0))
            delta = tgt - shares
            px0 = np.nan_to_num(exec_px)
            buy_px = px0 * (1 + slippage)
            sold_px = px0 * (1 - slippage)
            buy = float((np.clip(delta, 0, None) * buy_px).sum())
            sell = float((-np.clip(delta, None, 0) * sold_px).sum())
            fee = buy * fee_buy + sell * fee_sell
            cash += sell - buy - fee
            shares = shares + delta
            turnover_a[k] = 0.5 * (buy + sell) / nav_pre if nav_pre > 0 else 0.0
            fees_a[k] = fee
            trades_a[k] = delta.astype("float32")
            shares_a[k] = shares.astype("float32")
        nav_a[i] = cash + float((shares * np.nan_to_num(vl[i])).sum())
        nav_open_a[i] = cash + float((shares * np.nan_to_num(opv[i])).sum())
        cash_a[i] = cash

    nav = pd.Series(nav_a, index=days)
    nav_open = pd.Series(nav_open_a, index=days)
    cash_s = pd.Series(cash_a, index=days)
    nav_pre_s = pd.Series(nav_pre_a, index=exec_days)
    turnover = pd.Series(turnover_a, index=exec_days)
    fees = pd.Series(fees_a, index=exec_days)
    trades = pd.DataFrame(trades_a, index=exec_days, columns=cols)
    shares_hist = pd.DataFrame(shares_a, index=exec_days, columns=cols)

    if n_empty > max(2, len(exec_days) // 10):
        raise AssertionError(
            f"{n_empty}/{len(exec_days)} exec days had no tradable name -- price/mask table degraded by row, refusing")
    if n_empty:
        logger.warning(f"eng n_empty:{n_empty}/{len(exec_days)} "
                       f"turnover:{float(turnover.sum()):.4f} fees:{float(fees.sum()):.4f}")
    return {"nav": nav, "nav_open": nav_open, "nav_pre": nav_pre_s, "cash": cash_s,
            "turnover": turnover, "fees": fees, "trades": trades, "shares": shares_hist}


def run_all_layers(W, panels, rdates, **kw):
    out = []
    for k, w in enumerate(W):
        if float(np.abs(w.to_numpy()).sum()) == 0.0:
            print(f"  ⚠ 层 {k}：目标权重全 0（空层）→ 净值将恒为 1 / 零成交")
        out.append(run_layer(w, panels, rdates, **kw))
    return out


def nav_frame(results: list[dict]) -> pd.DataFrame:
    """把 list[5] 账本拼成【5 列净值表 nav_df】（列名 = 层号 0..4），供 metrics 消费。

    results : run_all_layers(...) 的返回（list[5]，每层一个 dict）
    返回     : DataFrame(交易日 × 5)，列名 = 层号 —— 这就是下游统一称呼的 `nav_df`【二维】
               单层那一维在 res[L]["nav"] 里【一维】，两者别混（《数据契约》规则 6）
    """
    return pd.DataFrame({i: r["nav"] for i, r in enumerate(results)})


def main():
    import time
    from backtest.panel import load_panels
    from backtest.assemble import rebalance_dates
    from backtest.weights import weights_for_factor

    t0 = time.time()
    panels = load_panels()
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index) if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    res = run_all_layers(W, panels, rdates)
    # 【二维】5 列净值表（列名 = 层号）；单层是 res[L]["nav"]【一维】
    nav_df = nav_frame(res)
    print(
        f"面板 {panels['hfq_close'].shape} ｜ 调仓日 {len(rdates)} ｜ 5 层 ｜ 用时 {time.time() - t0:.1f}s")
    print()
    print(f"{'层':<4}{'期末净值':>12}{'年化':>10}{'最大回撤':>12}{'累计换手':>12}{'费用合计':>12}")
    for L, r in enumerate(res):
        n = r["nav"].dropna()
        ann = n.iloc[-1] ** (244 / len(n)) - 1
        dd = float((n / n.cummax() - 1).min())
        print(
            f"{L:<4}{n.iloc[-1]:>12.4f}{ann:>10.2%}{dd:>12.2%}{r['turnover'].sum():>12.2f}{r['fees'].sum():>12.5f}")
    ls = nav_df[4] / nav_df[0] - 1
    print()
    print(
        f"多空（比值法：层4 / 层0）= {ls.iloc[-1]:.2%} ｜ 年化 {(1 + ls.iloc[-1]) ** (244 / len(ls)) - 1:.2%}")
    print("注意：含佣金 + 印花税 + 5bp 滑点；无融券费 / 无整手 / 无涨跌停；换手是一侧口径 0.5×Σ|Δw|")


if __name__ == "__main__":
    main()
