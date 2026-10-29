from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from backtest.panel import derive_flags

_PANEL_SNAP = {}


def _panel_arrays(panels, cols):
    """把几张大面板表转成 numpy 并且【当场验一遍】—— 整个进程只做一次（以前是每个因子都重做）。

    为什么要验：吃过一次亏 —— 拿到一份不完整的快照，nav_pre 算小了，
    最后换手率算出来只有 5（正常的应该是 100 上下），而且一点都不报错（PITFALLS #24）。
    顺带缓存还省掉了 18 次里的 17 次复制。
    """
    key = (id(panels["hfq_close"]), id(panels["hfq_open"]),
           id(panels["can_trade"]), id(panels["reason"]), tuple(cols))
    if _PANEL_SNAP.get("key") == key:
        return (_PANEL_SNAP["op"], _PANEL_SNAP["vl"], _PANEL_SNAP["ct"],
                _PANEL_SNAP["cb"], _PANEL_SNAP["cs"], _PANEL_SNAP["_rs"])

    # 按【标签】对齐（等价老的 panels[...][cols]）：先拿到列的整数位置，再用 numpy 选列
    # 按标签取列：W 里如果有面板不存在的股票代码，这里会直接抛 KeyError（不许静默缺列）
    _ci = panels["hfq_close"].columns.get_indexer(cols)
    _missing = [c for c, q in zip(cols, _ci) if q < 0]
    if _missing:
        raise KeyError(f"W 里有面板不存在的股票代码: {_missing[:5]}"
                       f"{' 等 ' + str(len(_missing)) + ' 个' if len(_missing) > 5 else ''}")
    op = np.ascontiguousarray(panels["hfq_open"].to_numpy()[:, _ci])
    _cl = np.ascontiguousarray(panels["hfq_close"].to_numpy()[:, _ci])
    _ct = np.ascontiguousarray(panels["can_trade"].to_numpy()[:, _ci])
    _rs = np.ascontiguousarray(panels["reason"].to_numpy()[:, _ci])
    _fl = derive_flags(_rs)
    cb = np.ascontiguousarray(_fl["can_buy"])
    cs = np.ascontiguousarray(_fl["can_sell"])

    _idx = np.where(~np.isnan(_cl), np.arange(_cl.shape[0])[:, None], 0)
    np.maximum.accumulate(_idx, axis=0, out=_idx)
    # 用 numpy 做 ffill（等价 pandas 的 .ffill()，但循环里不允许出现 pandas）
    vl = np.take_along_axis(_cl, _idx, axis=0)
    ct = np.ascontiguousarray(_ct)

    assert op.shape == vl.shape == ct.shape, "panel arrays have inconsistent shapes"
    if ct.shape[1] > 1000:
        _m = float(ct.mean())
        assert 0.4 < _m < 0.95, f"can_trade True ratio {_m:.3f} out of range -- snapshot degraded"
        assert float(np.isfinite(op).mean()
                     ) > 0.3, "hfq_open mostly non-finite -- snapshot degraded"
        assert float(np.isfinite(vl).mean(
        )) > 0.3, "hfq_close ffill mostly non-finite -- snapshot degraded"

    _PANEL_SNAP.update(key=key, op=op, vl=vl, ct=ct, cb=cb, cs=cs, _rs=_rs)
    return op, vl, ct, cb, cs, _rs


def run_layer(W_layer, panels, rdates, *, fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005, init_cash=1.0):
    days = panels["hfq_close"].index
    cols = W_layer.columns
    # 入口全部走 numpy（2026-10-13 改）：不再用 pandas 的 `[cols]` / `.eq(True)` / `.copy()` 链
    # —— 那条链被实测到会**偶发**产出"形状正确、dtype 正确、但数据全 False"的表（非确定性缺陷）
    # dtype 跟随面板（真实面板 float32 / 合成面板 float64）
    # 入口一律走 numpy，不用 pandas 的 [cols] / .ffill() 链 ——
    # 那条链被实测到过「形状对、dtype 对、但数值全错」的偶发情况（PITFALLS #23/#24），
    # 表现是 nav_pre 偏小，最后换手率算出来只有 5 左右（正常应该是 100 上下）
    op, vl, ct, cb, cs = _panel_arrays(panels, cols)
    # 开盘估值价：今天有开盘价就用开盘价，没有（停牌/缺数据）就退到前一交易日的收盘价
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
        # T-2.3：两个方向的掩码也各查一条（涨停只封买 ⇒ can_buy 略小于 can_sell ✓）
        _buy_r = float(cb[_rows].mean())
        _sell_r = float(cs[_rows].mean())
        assert 0.4 < _buy_r < 0.95, f"执行日 can_buy 占比 {_buy_r:.3f} 越界（健康 ≈0.728）—— reason 表退化？"
        assert 0.4 < _sell_r < 0.95, f"执行日 can_sell 占比 {_sell_r:.3f} 越界（健康 ≈0.729）—— reason 表退化？"
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

    # 一次性把权重拍成快照 —— 之后整个循环里一个 pandas 对象都不许碰
    # 优先用 build_target_weights 直接递过来的 ndarray：零拷贝，
    # 所以不存在「复制到一半被读到」的窗口（那个症状是换手率 < 20）
    trades_a = np.zeros((n_ex, n_cols),    dtype="float32")
    shares_a = np.zeros((n_ex, n_cols),    dtype="float32")
    blocked_buy_a = np.zeros((n_ex, n_cols),  dtype="float32")
    blocked_sell_a = np.zeros((n_ex, n_cols), dtype="float32")

    Wn = getattr(W_layer, "attrs", {}).get("np")
    if Wn is not None:
        # 来自 build_target_weights 的真权重 ⇒ 每一行的和必须接近 1（满仓或空层）
        Wn = np.ascontiguousarray(Wn)
        _rs = np.abs(Wn).sum(axis=1)
        assert float(_rs.min()) > 0.9, (
            f"W has a bad row (min row sum {float(_rs.min()):.4f}) -- refusing to run the engine")
    else:
        # 手工造的权重（测试 / 合成夹具）：允许有空层，所以不做行和校验
        Wn = np.nan_to_num(W_layer.to_numpy(
            dtype="float64", copy=True), nan=0.0)
    rd_row = {d: i for i, d in enumerate(W_layer.index)}
    ex_row = {d: k for k, d in enumerate(exec_days)}

    for i, dt in enumerate(days):
        if i in exec_pos:
            k = ex_row[dt]
            w = Wn[rd_row[exec_pos[i]]]

            # ① 两个方向各自的成交许可：涨停只封买、跌停只封卖（T-2.3 的核心）
            #    b_ok/s_ok 各自独立，所以同一天可以"能卖不能买"或"能买不能卖" ✓
            b_ok = cb[i] & (op[i] > 0.0)
            s_ok = cs[i] & (op[i] > 0.0)
            tradable = ct[i] & (op[i] > 0.0)      # 只用来数 n_empty（"今天一只都动不了"）
            if not tradable.any():
                n_empty += 1

            # ② 估值：opv = 有开盘价就用开盘价，缺了退前一日收盘（停牌日靠这个）
            val_px = np.nan_to_num(opv[i])
            nav_pre = cash + float((shares * val_px).sum())
            nav_pre_a[k] = nav_pre

            # ③ 目标股数：用【今日开盘价】算（px0 兜 1.0 只是防 0 除，不会真的按 1 元成交）
            px0 = np.nan_to_num(op[i])
            want = nav_pre * w / np.where(px0 > 0.0, px0, 1.0)
            delta = want - shares

            # ④ 分方向裁掉：卖不掉的保持原样（d_sell=0），买不到的放弃（d_buy=0）
            #    ★ delta 在这里被劈成两半 —— 这是整个改动唯一的新语义
            d_sell = np.where((delta < 0.0) & s_ok, delta, 0.0)     # 负数
            d_buy = np.where((delta > 0.0) & b_ok, delta, 0.0)      # 正数

            blocked_buy_a[k] = np.where((delta > 0.0) & ~b_ok, delta, 0.0)
            blocked_sell_a[k] = np.where((delta < 0.0) & ~s_ok, -delta, 0.0)

            # ⑤ 先卖：卖出所得【当天】就能拿去买（T+1 只限制提现，不限制当天买入）
            #    注意 cash_avail 要扣【卖出费】—— 那是净到手，不是毛收入 ✓
            sell_val = float((-d_sell * px0 * (1.0 - slippage)).sum())
            cash_avail = cash + sell_val - sell_val * fee_sell

            # ⑥ 再买：钱不够就【全组合等比缩量】，不在别处补仓
            #    ★ need 必须把【买入手续费】也算进去（= 滑点 × (1+fee_buy)），
            #      否则 scale 会把现金精确花到 0，随后 fee 一扣就透支
            #      （实测：漏了这一步 ⇒ cash = -0.000641，被下面那条断言当场抓住 ✓）
            buy_coef = (1.0 + slippage) * (1.0 + fee_buy)
            need = float((d_buy * px0 * buy_coef).sum())
            scale = min(1.0, cash_avail / (need + 1e-12)
                        ) if need > 0.0 else 1.0
            d_buy *= scale

            # ⑦ 结算：现金与持仓同时推进，然后当场验"没透支"
            buy_val = float((d_buy * px0 * (1.0 + slippage)).sum())
            fee = buy_val * fee_buy + sell_val * fee_sell
            cash += sell_val - buy_val - fee
            shares = shares + d_sell + d_buy
            assert cash >= -1e-9, (
                f"现金透支 {cash:.6f}（scale={scale:.4f} need={need:.4f} "
                f"avail={cash_avail:.4f} buy_coef={buy_coef:.6f}）"
                f"—— 缩量或掩码算错了")

            turnover_a[k] = 0.5 * (buy_val + sell_val) / \
                nav_pre if nav_pre > 0 else 0.0
            fees_a[k] = fee
            trades_a[k] = (d_sell + d_buy).astype("float32")
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
    block_buy = pd.DataFrame(blocked_buy_a, index=exec_days, column=cols)
    block_sell = pd.DataFrame(blocked_sell_a, index=exec_days, column=cols)

    if n_empty > max(2, len(exec_days) // 10):
        raise AssertionError(
            f"{n_empty}/{len(exec_days)} exec days had no tradable name -- price/mask table degraded by row, refusing")
    if n_empty:
        logger.warning(f"eng n_empty:{n_empty}/{len(exec_days)} "
                       f"turnover:{float(turnover.sum()):.4f} fees:{float(fees.sum()):.4f}")
    return {"nav": nav, "nav_open": nav_open, "nav_pre": nav_pre_s, "cash": cash_s,
            "turnover": turnover, "fees": fees, "trades": trades, "shares": shares_hist,
            "block_buy": block_buy, "block_sell": block_sell}


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
