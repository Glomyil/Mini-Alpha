from __future__ import annotations
import numpy as np
import pandas as pd


def run_layer(W_layer, panels, rdates, *, fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005, init_cash=1.0):
    days = panels["hfq_close"].index
    cols = W_layer.columns
    # 入口全部走 numpy（2026-10-13 改）：不再用 pandas 的 `[cols]` / `.eq(True)` / `.copy()` 链
    # —— 那条链被实测到会**偶发**产出"形状正确、dtype 正确、但数据全 False"的表（非确定性缺陷）
    op = panels["hfq_open"][cols].to_numpy()            # dtype 跟随面板（真实面板 float32 / 合成面板 float64）
    vl = panels["hfq_close"][cols].ffill().to_numpy()   # 每日收盘估值价
    pv = np.empty_like(vl)                              # ＝ px_val.shift(1)，dtype 与面板一致
    pv[0] = np.nan
    pv[1:] = vl[:-1]
    opv = np.where(np.isnan(op), pv, op)                # 开盘估值价（缺则截至上一交易日收盘）
    ct = panels["can_trade"][cols].to_numpy()           # bool

    # —— 健康度断言（2026-10-13）：把"数据退化"变成响亮失败，而不是静默算错 ——
    assert ct.size and ct.any(), f"can_trade 全 False（{ct.shape}）—— 数据退化，拒绝回测"
    if ct.shape[1] > 1000:                                             # 全市场面板才做区间校验（合成小面板会越界）
        _r = float(ct.mean())
        assert 0.4 < _r < 0.95, f"can_trade True 占比 {_r:.3f} 越界（健康值 ≈0.73）—— 数据可疑"
        _op = float((op > 0.0).mean())
        _vl = float((vl > 0.0).mean())
        assert _op > 0.3, f"hfq_open 正价占比 {_op:.3f} 过低（健康值 ≈0.75）—— 价格表退化，拒绝回测"
        assert _vl > 0.3, f"hfq_close 正价占比 {_vl:.3f} 过低（健康值 ≈0.75）—— 价格表退化，拒绝回测"

    pos = days.get_indexer(pd.DatetimeIndex(rdates))
    exec_pos = {}
    for p, dt in zip(pos, pd.DatetimeIndex(rdates)):
        if p >= 0 and p + 1 < len(days):
            exec_pos[p+1] = dt
    exec_days = [days[i]for i in sorted(exec_pos)]

    shares = pd.Series(0.0, index=cols)
    cash = float(init_cash)
    nav = pd.Series(np.nan, index=days, dtype="float64")
    nav_open = pd.Series(np.nan, index=days,
                         dtype="float64")        # 诊断：每日开盘时刻的账户值
    nav_pre_s = pd.Series(np.nan, index=exec_days,
                          dtype="float64")  # 诊断：成交前净值（只在执行日）
    cash_s = pd.Series(np.nan, index=days, dtype="float64")
    turnover = pd.Series(0.0, index=exec_days, dtype="float64")
    fees = pd.Series(0.0, index=exec_days, dtype="float64")
    trades = pd.DataFrame(0.0, index=exec_days, columns=cols, dtype="float32")
    shares_hist = pd.DataFrame(
        0.0, index=exec_days, columns=cols, dtype="float32")
    n_empty = 0            # 无票可成交的执行日数（按行退化的信号：全局均值可能仍正常）

    for i, dt in enumerate(days):
        if i in exec_pos:
            w = W_layer.loc[exec_pos[i]].fillna(0.0).to_numpy()
            tradable = ct[i] & (op[i] > 0.0)
            if not tradable.any():
                n_empty += 1
            exec_px = np.where(tradable, op[i], np.nan)
            val_px = np.nan_to_num(opv[i])
            nav_pre = cash + float((shares.to_numpy() * val_px).sum())
            nav_pre_s.loc[dt] = nav_pre                  # 留痕：成交前净值（对账用）
            tgt = np.where(np.isnan(exec_px), shares.to_numpy(),
                           nav_pre*w/np.nan_to_num(exec_px, nan=1.0))
            delta = tgt - shares.to_numpy()
            px0 = np.nan_to_num(exec_px)
            buy_px = px0*(1+slippage)
            sold_px = px0*(1-slippage)
            buy = float((np.clip(delta, 0, None)*buy_px).sum())
            sell = float((-np.clip(delta, None, 0)*sold_px).sum())
            fee = buy*fee_buy + sell*fee_sell
            cash += sell-buy-fee
            shares = pd.Series(shares.to_numpy()+delta, index=cols)
            turnover.loc[dt] = 0.5*(buy+sell)/nav_pre if nav_pre > 0 else 0.0
            fees.loc[dt] = fee
            trades.loc[dt] = delta.astype("float32")
            shares_hist.loc[dt] = shares.to_numpy().astype("float32")
        nav.iloc[i] = cash + float((shares.to_numpy() * np.nan_to_num(vl[i])).sum())
        nav_open.iloc[i] = cash + float((shares.to_numpy() * np.nan_to_num(opv[i])).sum())
        cash_s.iloc[i] = cash
    # —— 按行健康度（2026-10-13）：全局均值正常也可能是"执行日那几行"退化 ——
    # 实测健康态：129 个执行日里 0 个"无票可成交"（可成交率 99.8%）；退化时会是全部或大部分
    if n_empty > max(2, len(exec_days) // 10):
        raise AssertionError(
            f"{n_empty}/{len(exec_days)} 个执行日无可成交标的 —— 价格/掩码表按行退化，拒绝交付结果")
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
