from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd

TRADING_DAYS = 244
REBALANCE_PERIODS_PER_YEAR = 12


def _returns(nav: pd.Series) -> pd.Series:
    """逐期收益。纯 numpy 显式写满，不借助 pct_change（见 _period_returns 的说明）。"""
    n_bad = int(nav.isna().sum())
    if n_bad:
        raise ValueError(f"nav 含 {n_bad} 个 NaN —— 不许静默 dropna，先回去修 engine")
    px = nav.to_numpy(dtype="float64", copy=True)
    out = np.full(px.size, np.nan, dtype="float64")
    out[1:] = px[1:] / px[:-1] - 1.0
    r = out[1:]
    # 只留 A（无 NaN）。长度由 out[1:] 结构保证；往返对 numpy 实现代数恒真（§4.4.0）。
    assert not np.isnan(r).any(), "收益里出现 NaN —— 净值被污染（PITFALLS #23）"
    return pd.Series(r, index=nav.index[1:], name=nav.name)


def ann_return(nav: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """年化收益。(末/首)^(trading_days/天数) − 1。

    trading_days 必须由调用方从 spec 传进来（yml 的 research.trading_days）。
    本模块的 TRADING_DAYS=244 只是 fallback —— 否则会出现「报告表头印 252、
    数字仍按 244 算」的两处并存（PITFALLS #19/#29）。
    """
    if len(nav) < 2:
        return float("nan")
    return float(nav.iloc[-1] / nav.iloc[0]) ** (trading_days / len(nav)) - 1.0


def ann_vol(nav: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """年化波动。trading_days 同上，从 spec 传。"""
    r = _returns(nav)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return float("nan")
    return float((nav / nav.cummax() - 1.0).min())


def sharpe(nav: pd.Series, rf: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    vol = ann_vol(nav, trading_days)
    if not np.isfinite(vol) or vol < 1e-12:
        return float("nan")
    return float((ann_return(nav, trading_days) - rf) / vol)


def turnover_total(turnover: pd.Series) -> float:
    return float(turnover.sum())


def turnover_ann(turnover: pd.Series, nav: pd.Series,
                 trading_days: int = TRADING_DAYS) -> float:
    years = len(nav) / trading_days
    return float(turnover.sum() / years) if years > 0 else float("nan")


def fee_total(fees: pd.Series) -> float:
    return float(fees.sum())


def _rank_avg(v):
    """把值换成排名（1..n），并列取平均 —— 和 pandas 的 rank(method="average") 同义。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman_rank(a, b):
    """两个序列的 Spearman（秩相关），零依赖。任一序列是常数时返回 None。

    ⚠️ 不要用 pandas 的 .corr(method="spearman")：那个内部要 scipy，
    而 requirements.txt 里没有 scipy ⇒ CI 上会 ModuleNotFoundError ✗
    Spearman 的定义就是「先换排名，再算 Pearson」✓
    """
    try:
        return float(statistics.correlation(_rank_avg(list(a)), _rank_avg(list(b))))
    except statistics.StatisticsError:
        return None      # 有常数序列 ⇒ 相关系数无定义（pandas 那边返回 nan）


def rank1d(x, ties="first"):
    """自己算排名（1..n），不调 pandas。

    并列怎么处理由 ties 决定：
      ties="first"   ⇒ 等价 pandas 的 rank(method="first")   —— 分层用这个（必须能分出先后）
      ties="average" ⇒ 等价 pandas 的 rank(method="average") —— 教科书 Spearman 用这个（算 IC）
    """
    x = np.asarray(x, dtype="float64")
    order = np.argsort(x, kind="stable")
    r = np.zeros(x.size, dtype="float64")   # zeros, not empty (PITFALLS #24)
    r[order] = np.arange(1, x.size + 1, dtype="float64")
    if ties == "average":
        xs = x[order]
        new = np.empty(xs.size, dtype=bool)
        new[0] = True
        # run boundaries (values already sorted)
        new[1:] = xs[1:] != xs[:-1]                 # 找连续段的边界（xs 已经排好序）
        # group id per sorted position
        gid = np.cumsum(new) - 1
        cnt = np.bincount(gid)
        # sum of ranks inside each group
        s = np.bincount(gid, weights=r[order])
        # mean rank per element (vectorized)
        r[order] = (s / cnt)[gid]

    return r


_SNAP = {}   # panel snapshot memo: key -> (C, K)


def rank_ic(factor: pd.DataFrame, panels: dict, rdates, min_names: int = 30) -> pd.Series:
    close, ok = panels["hfq_close"], panels["can_trade"]
    rdates = list(rdates)
    # --- 纯 numpy 的核心：先拍一次快照，之后全部用整数行号取数据（PITFALLS #24）---
    # 循环里不用 pandas 的 .loc / .corr —— 逐日抽行曾经是最后一个"偶发读到未写入页"
    # 导致横截面被污染的地方（症状是某一天的 Spearman 突然算不出来）
    cols = close.columns
    pos = {d: i for i, d in enumerate(close.index)}
    # 把大面板快照缓存起来：每个因子看到的都是同一份，重复转 numpy 纯属浪费（PITFALLS #24）
    key = (id(close), id(ok), close.shape)
    if _SNAP.get("key") != key:
        _SNAP["key"] = key
        _SNAP["C"] = np.ascontiguousarray(
            close.to_numpy(dtype="float64", copy=True))
        _SNAP["K"] = np.ascontiguousarray(
            ok.reindex(columns=cols).fillna(False).to_numpy(dtype=bool, copy=True))
    C, K = _SNAP["C"], _SNAP["K"]
    assert list(factor.columns) == list(
        cols), "factor columns must match panel columns"
    F = np.ascontiguousarray(factor.to_numpy(dtype="float64", copy=True))
    ri = np.array([pos.get(d, -1) for d in rdates], dtype=np.int64)

    out = {}
    for i in range(len(rdates) - 1):
        a, b = int(ri[i]), int(ri[i + 1])
        if a < 0 or b < 0:
            continue
        fv, cv, nv, kv = F[a], C[a], C[b], K[a]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            rv = nv / cv - 1.0
        good = kv & np.isfinite(fv) & np.isfinite(rv)
        k = int(good.sum())
        if k < min_names:
            continue
        rx = rank1d(fv[good], ties="average")
        ry = rank1d(rv[good], ties="average")
        a = rx - rx.mean()
        b = ry - ry.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        val = float((a * b).sum() / den) if den > 0 else float("nan")
        if not np.isfinite(val):
            raise AssertionError(
                f"{rdates[i].date()} Spearman undefined (constant cross-section / all-NaN) "
                f"-- data degraded, refusing to report (PITFALLS #23)")
        out[rdates[i]] = val
    return pd.Series(out, name="rank_ic", dtype="float64")


def ic_summary(ic: pd.Series,
               periods_per_year: int = REBALANCE_PERIODS_PER_YEAR, lags: int | None = None) -> dict:
    # valid periods (DEBT-013: len(ic) used to count NaN periods too)
    n = int(ic.notna().sum())        # 有效期数（DEBT-013：以前用 len(ic)，把 NaN 的期也算进去了）
    n_bad = int(len(ic) - n)             # invalid periods
    if n == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_ir": float("nan"), "ic_ir_nw": float("nan"),
                "ic_pos_rate": float("nan"), "nw_lags": 0,
                "n": 0, "n_pos": 0, "n_bad": n_bad}
    ic = ic.dropna()
    sd = float(ic.std(ddof=1))
    # IC>0 胜率：有均值没稳定性会飘，胜率才说明「方向稳不稳」。
    # 严格 > 0（IC 恰好 0 不算正）——和券商研报口径一致。
    # n_pos 直接返回，不让调用方拿比率乘回去四舍五入（那会偶尔差 1 期）。
    n_pos = int((ic > 0).sum())
    # Newey-West：IC 序列同样有自相关（风格切换有惯性）⇒ 朴素 IC-IR 也会失真
    _nw_lag = nw_lags(n) if lags is None else max(0, int(lags))
    t_nw, _p_nw = t_stat_nw(ic.to_numpy(dtype="float64"), lags=_nw_lag)
    return {"ic_mean": float(ic.mean()),
            "ic_std": sd,
            # ⚠️ 年化粒度必须由调用方按 freq 传：写死 12 的话 freq=W-FRI 时
            #    banner 印「IC-IR 用 52 期/年」而数字按 √12 算 —— IR 少报 √(52/12)≈2 倍。
            #    退化的判法照抄 sharpe() 的绝对阈值，不用 if sd（浮点残差会让它印 1e16）。
            "ic_ir": (float(ic.mean() / sd * np.sqrt(periods_per_year))
                      if sd >= 1e-12 else float("nan")),
            # 2026-11-11（T-3.7）：IC-IR 的 Newey-West 版。
            #   恒等式：IC-IR = t * sqrt(ppy/n)，其中 t = mean/(sd/sqrt(n))
            #   ⇒ NW 版就是把 t 换成 t_nw。所以它和朴素版【同一量纲】，可以直接比。
            #   ⚠️ 别在这里重写一遍 HAC 公式 —— 用 t_stat_nw 那个已被 statsmodels 对拍过的。
            "ic_ir_nw": (float(t_nw * math.sqrt(periods_per_year / n))
                         if np.isfinite(t_nw) else float("nan")),
            "ic_pos_rate": float(n_pos / n),
            "nw_lags": _nw_lag,
            "n": n, "n_pos": n_pos, "n_bad": n_bad}


def _period_returns(nav: pd.Series, index) -> np.ndarray:
    """按给定 index 取净值，算逐期收益（纯 numpy，显式写满）。

    为什么不用 nav.reindex(index).pct_change()：
      pct_change 内部是 s / s.shift(1) - 1，属于 4.4.0 说的「部分填充的库操作」。
      窄表实测 200 次 0 异常，但没必要承担「依赖库行为」这个风险 —— 一行除法就能写满。

    只留一道断言（A 无 NaN）—— 见 §4.4.0 的「过度防御」裁决表：
    长度断言结构上恒真、往返断言对 numpy 实现代数恒真，都不能在正常情况下为假。
    """
    px = nav.reindex(index).to_numpy(dtype="float64", copy=True)

    out = np.full(px.size, np.nan, dtype="float64")
    out[1:] = px[1:] / px[:-1] - 1.0
    r = out[1:]                       # 等价 pct_change().dropna()，但没借助 shift

    n_nan = int(np.isnan(r).sum())
    assert n_nan == 0, (
        f"逐期收益有 {n_nan} 个 NaN（共 {r.size}）—— "
        f"rdates 里有 {len(index)} 个日期，其中有的不在 nav.index 上（reindex 产生了 NaN），"
        f"或净值某期缺失")
    return r


def _ann_from_periods(nav_np: np.ndarray, periods_per_year: int) -> float:
    """把【调仓频率】的净值序列年化。

    ⚠️ 不能用 ann_return —— 它假定 nav 是【每日】序列（TRADING_DAYS/len(nav)）。
       多空序列是【每期】的。实测（128 期、每期 +0.5%，总收益 89.35%，约 10.7 年）：
         ann_return 口径  → +234.52%   ← 把 10.7 年当成 244 天
         periods_per_year →   +6.17%   ← 正确
       复核 (1+6.17%)^10.67 − 1 = 89.35% = 总收益 ✓

    nav_np : 长度 = 期数 + 1（第 0 个点是起点 1.0）
    """
    if nav_np.size < 2:
        return float("nan")
    n_period = nav_np.size - 1
    return float(nav_np[-1] / nav_np[0]) ** (periods_per_year / n_period) - 1.0


def _norm_p(t: float) -> float:
    """双尾 p 值，正态近似（零依赖，用 math.erf）。

    df=128 时与 t 分布差 < 0.003（实测 t=1.98 ⇒ 0.0477 vs 真实 0.0500）。
    df < 10 时会偏松 ⇒ 报告里必须注明「正态近似」。
    """
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def nw_lags(n: int) -> int:
    """Newey-West 的 lag 经验法则：floor(4*(T/100)^(2/9))。

    常见样本量下：T=130 ⇒ 4 期；T=552 ⇒ 5 期；T=1000 ⇒ 6 期（实测值，有测试钉着）。
    为什么要有它：lag 是个【研究选择】，不写死一个默认值就等于把选择藏起来。
    报告里必须把它印出来（用了几个 lag）。
    """
    n = int(n)
    if n < 2:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_se(series, lags: int | None = None, use_correction: bool = True) -> float:
    """均值的 Newey-West（Bartlett 核）HAC 标准误。

    朴素 SE 假设「每期独立」；收益/IC 有自相关时那是错的 ——
    正自相关会让真实 SE 远大于朴素值 ⇒ 朴素 t 被系统性高估。

    公式（Bartlett 权重，Newey-West 1987）：
        g_j = (1/T) Σ_{t>=j} (x_t - x̄)(x_{t-j} - x̄)
        S   = g_0 + 2 Σ_{j=1..L} (1 - j/(L+1)) g_j
        SE  = sqrt(S / T)

    ⚠️ 三个容易写错的点：
      ① 归一化用 1/T（不是 1/(T-j)）；
      ② Bartlett 权重是 (1 - j/(L+1))，写反成 j/(L+1) 会让长 lag 权重更大；
      ③ lags=0 必须【逐位退化】成朴素 SD/√n —— 这条有测试守着。

    lags 传 None ⇒ 用 nw_lags(T) 的经验法则；显式传 0 才是「不做 HAC 修正」。

    use_correction：乘一个小样本因子 T/(T-1)（我们只估 1 个参数=均值）。
      这不是可选项 —— 不乘的话 lags=0 会得到 sqrt((T-1)/T) 倍的偏小值，
      和报告里那列朴素 SE 对不上，读者会以为两列有一列算错了。
      statsmodels 的 cov_hac(use_correction=True) 也是这么做的。
    """
    r = np.asarray(series, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return float("nan")
    if lags is None:
        lags = nw_lags(n)
    L = max(0, int(lags))
    e = r - r.mean()
    g0 = float((e * e).sum()) / n
    S = g0
    for j in range(1, min(L, n - 1) + 1):
        gj = float((e[j:] * e[:-j]).sum()) / n
        S += 2.0 * (1.0 - j / (L + 1.0)) * gj
    if S <= 0:            # 负自相关极强时 Bartlett 也可能给出 <=0（非常罕见）
        return float("nan")
    if use_correction and n > 1:
        S *= n / (n - 1.0)      # 小样本修正 ⇒ lags=0 时正好还原 SD/√n
    return float(math.sqrt(S / n))


def t_stat_nw(series, lags: int | None = None) -> tuple[float, float]:
    """t_stat 的 Newey-West 版：t = mean / SE_hac，p 用同一个正态近似重算。

    ⚠️ p 必须由【新的 t】重算。直接沿用朴素 p 是最常见的错 ——
       那样 t 变了 p 没变，报告里两列会自相矛盾。

    lags=0 时与 t_stat 完全逐位一致（有测试守着）。
    """
    r = np.asarray(series, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return float("nan"), float("nan")
    if np.ptp(r) == 0.0:
        return float("nan"), float("nan")
    se = newey_west_se(r, lags=lags)          # lags=0 时 newey_west_se 自己会退化成朴素 SE
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan")
    t = float(r.mean() / se)
    return t, _norm_p(t)


def t_stat(series) -> tuple[float, float]:
    """配对 t 与双尾 p：t = mean/(std/√n)。

    ⚠️ series 必须是【调仓频率】的差值，不能传日频 ——
       日收益自相关严重，会把 n 虚假放大 ⇒ t 虚高好几倍。

    返回 (nan, nan) 的两种情况：样本 < 3 期、标准差为 0（全等值）。
    """
    r = np.asarray(series, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return float("nan"), float("nan")
    # 全等值 ⇒ 方差为 0 ⇒ t 无定义。用极差判，不用 sd == 0：
    # 0.1 在二进制里不精确，[0.1,0.1,0.1] 的 np.std 算出 1.7e-17 而不是 0，
    # 那样 sd == 0.0 判不出来，t 会变成 1e16 这种垃圾数印进报告。
    if np.ptp(r) == 0.0:
        return float("nan"), float("nan")
    sd = float(r.std(ddof=1))
    t = float(r.mean() / (sd / np.sqrt(n)))
    return t, _norm_p(t)


def ls_summary(nav_df: pd.DataFrame, rdates, *, borrow_cost: float = 0.08,
               periods_per_year: int = 12,
               trading_days: int = TRADING_DAYS) -> dict:
    """多空诊断：三个口径，各带年化 + 累计 + 假设说明。

    ⚠️ 这里【没有】比值法 —— 那是两个组合净值之比（nav_l/nav_s − 1），不是多空收益。
       层0 越难看它越好看，会让人误以为多空赚了 350%。

    ① 层差(pp)     ann_return(层4) − ann_return(层0)（两个都是日频，各算各的）
       ⇒ 因子单调性强度。不做空，不涉及融券。【没有「累计」这个概念】
    ② 多空年化     ∏(1 + r₄ − r₀) 按 periods_per_year 年化
       ⇒ 每期再平衡、零成本做空的市场中性策略理论上限
    ③ 多空净年化   ② − borrow_cost
       ⇒ 扣融券费后的真实水平（A 股约 8~10%/年）

    periods_per_year 必须从 spec 来（freq 是唯一源头），不能写死 12。
    """
    assert isinstance(nav_df, pd.DataFrame), \
        f"ls_summary 要的是 5 列净值表(nav_df)，收到的是 {type(nav_df).__name__}"
    assert nav_df.shape[1] >= 2, f"nav_df 至少 2 列（层0 和层4），实际 {nav_df.shape[1]}"

    nl = nav_df[nav_df.columns[-1]]
    ns = nav_df[nav_df.columns[0]]

    # ② 多空：按【调仓频率】取差值再复利
    #    不能日频 —— 日频再平衡是另一个你做不到的策略（要每天调多空头寸）
    rl = _period_returns(nl, rdates)
    rs = _period_returns(ns, rdates)
    diff = rl - rs

    # 复利：起点 1.0 + 每期累乘（纯 numpy，不用 pandas 的 cumprod 链，同 4.4.0）
    ls_nav = np.concatenate([[1.0], np.cumprod(1.0 + diff)])

    ls_ann = _ann_from_periods(ls_nav, periods_per_year)
    # 这里【不写往返断言】：ls_nav[-1] 就是 np.prod(1+diff)，字面上 prod 和自己比。
    # 真正的守卫在 _period_returns 的 A（收益无 NaN），上游已经拦过一次。

    t, p = t_stat(diff)
    # 2026-11-11（T-3.7）：Newey-West 版并列。多空收益序列有自相关（行业轮动、
    #   风格切换都有惯性）⇒ 朴素 t 被系统性高估。两列都印，让读者自己看差多少。
    _nw_lag = nw_lags(len(np.asarray(diff, dtype='float64')[np.isfinite(diff)]))
    t_nw, p_nw = t_stat_nw(diff, lags=_nw_lag)

    # ④ 多空夏普 / 最大回撤（2026-11-04 加）
    #    夏普 = 年化收益 / 年化波动；年化波动 = 每期标准差 × √N（是乘不是除 ——
    #    波动按 √N 放大，收益按复利年化，两者不能混）。
    #    每期收益恒定时标准差 = 0 ⇒ 夏普无定义，必须 NaN（不许 inf）——
    #    和 sharpe() 对「净值恒定」的处理保持同一条纪律。
    #   ⚠️ 判断「退化」不能只写 vol > 0 —— 恒定序列的 sd 是 1e-17 级浮点残差，
    #      会算出 3.2e14 的垃圾夏普。这里【照抄 sharpe() 的绝对阈值 1e-12】，
    #      不另发明一套：同一个项目里同一个概念只能有一种判法。
    _sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    _ann_vol = _sd * math.sqrt(periods_per_year)
    _sharpe = (float("nan")
               if (not np.isfinite(_ann_vol) or _ann_vol < 1e-12)
               else ls_ann / _ann_vol)
    # 最大回撤直接用 ls_nav（已经是复利累计），复用 max_drawdown 不用另写
    _mdd = max_drawdown(pd.Series(ls_nav))

    return {
        "层差(pp)":   float(ann_return(nl, trading_days) - ann_return(ns, trading_days)),
        "多空年化":   ls_ann,
        "多空累计":   float(ls_nav[-1] - 1.0),
        "多空净年化": ls_ann - borrow_cost,
        "多空t":      t,
        "多空p":      p,
        "多空t_nw":   t_nw,
        "多空p_nw":   p_nw,
        "nw_lags":    _nw_lag,
        "多空夏普":   _sharpe,
        "多空最大回撤": _mdd,
        "期数":       int(diff.size),
    }


def layer_table(nav_df: pd.DataFrame, results=None, rf: float = 0.0,
                trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    assert isinstance(nav_df, pd.DataFrame), \
        f"layer_table 要的是 5 列净值表(nav_df)，收到的是 {type(nav_df).__name__}（单层净值是一维 Series，别直接传）"

    rows = []
    for L in nav_df.columns:
        nav = nav_df[L]
        if nav.isna().any():
            raise ValueError(f"层 {L} 的 nav 含 NaN —— 不许静默 dropna")
        row = {"层": L,
               "期末净值": float(nav.iloc[-1]),
               "年化": ann_return(nav, trading_days),
               "波动": ann_vol(nav, trading_days),
               "最大回撤": max_drawdown(nav),
               "夏普": sharpe(nav, rf=rf, trading_days=trading_days)}
        if results is not None:
            row["累计换手"] = turnover_total(results[L]["turnover"])
            # 每期单边换手 = 累计 ÷ 期数。券商研报印的是【每期】那个数，
            # 只印累计会被读成"每期换手 105%"（容易误读）。
            _n_period = max(len(results[L]["turnover"]), 1)
            row["每期换手"] = row["累计换手"] / _n_period
            # ⚠️ 只含佣金/印花税。滑点被吃进成交价，体现在 nav 里，不记在这列
            #（engine.py:185 buy_val = …*(1+slippage)，所以滑点成本在 cash 上就扣过了）。
            row["佣金合计（不含滑点）"] = fee_total(results[L]["fees"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("层")


def _blocked_pp(res, panels, periods_per_year: int):
    """成交失败造成的年化损失（pp）。

    lost/nav 是【每期】的相对损失 ⇒ 乘 periods_per_year 才是年化。
    periods_per_year【必须由调用方按 spec.freq 显式传】（月频 12 / 季频 4 / 周频 52）。
    这里【故意不给默认值】：给了 12 就等于把月频假设藏起来 —— 改 freq 之后
    「一年少赚多少 pp」会静默地仍按月频算（PITFALLS #19「默认值藏口径」）。
    同 rebalance_dates(index, rule) 的纪律：忘传就 TypeError，不要静默。

    ⚠️ 这次定位到的根因（2026-11-03，PITFALLS #23）：
    pandas 的【宽表二元算术】（DataFrame / DataFrame）在全量面板
    （553x5476 ≈ 300 万元素）下偶发返回【常数垃圾数组】—— 全 0 或全 1，
    形状对、dtype 对、但数据全错。实测失败率：
      553x5476 float32  2/200 = 1.0%
      553x5476 float64  1/200 = 0.5%
      100x5476 / 553x500 / 100x500 / 50x50  全部 0/200
    ⇒ 规模依赖（与 PITFALLS #24 标题一致）。numpy 做同一个除法 200/200 稳定。
    所以这里连 hc.shift(-1)/hc 都不用了 —— 全部 numpy 显式写满。
    """
    ex = res["blocked_buy"].index
    hc = panels["hfq_close"].ffill().reindex(ex).to_numpy(dtype="float64")
    px = panels["hfq_open"].reindex(ex).to_numpy(dtype="float64")
    bb = res["blocked_buy"].to_numpy(dtype="float64")
    bs = res["blocked_sell"].to_numpy(dtype="float64")
    nav = res["nav_pre"].reindex(ex).to_numpy(dtype="float64")

    # nxt[t] = hc[t+1]/hc[t] - 1；最后一行没有下一期 => 显式预填 NaN
    nxt = np.full(hc.shape, np.nan, dtype="float64")
    nxt[:-1] = hc[1:] / hc[:-1] - 1.0

    # nansum / nanmean 对齐 pandas 的 skipna=True 语义
    lost = (np.nansum(bb * px * nxt, axis=1)
            - np.nansum(bs * px * nxt, axis=1))
    return float(np.nanmean(lost / nav)) * periods_per_year * 100


def equal_weight_nav(panels):
    r = panels["ret_1d_hfq"].where(panels["can_trade"])
    daily = r.mean(axis=1).fillna(0.0)
    return (1.0 + daily).cumprod()


def main():
    import time
    from backtest.panel import load_panels
    from backtest.assemble import build_features, rebalance_dates
    from backtest.factors import FACTORS
    from backtest.weights import weights_for_factor
    from backtest.engine import run_all_layers, nav_frame

    t0 = time.time()
    panels = load_panels()
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index, rule="ME") if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    res = run_all_layers(W, panels, rdates)
    nav_df = nav_frame(res)

    print(
        f"面板 {panels['hfq_close'].shape} ｜ 调仓日 {len(rdates)} ｜ 用时 {time.time() - t0:.1f}s")
    print()
    print(layer_table(nav_df, res).round(4).to_string())
    print()
    print("多空（三种定义）：", {k: f"{v:.2%}" for k,
          v in ls_summary(nav_df, rdates).items()})

    spec = FACTORS["rev_20d"]
    val = spec["fn"](build_features(panels, ["rev_20d"]), {}) * spec["sign"]
    print("IC：", {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in ic_summary(rank_ic(val, panels, rdates)).items()})


if __name__ == "__main__":
    main()
