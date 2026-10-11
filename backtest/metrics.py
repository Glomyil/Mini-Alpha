from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 244
REBALANCE_PERIODS_PER_YEAR = 12


def _returns(nav: pd.Series) -> pd.Series:
    n_bad = int(nav.isna().sum())
    if n_bad:
        raise ValueError(f"nav 含 {n_bad} 个 NaN —— 不许静默 dropna，先回去修 engine")
    return nav.astype("float64").pct_change(fill_method=None).dropna()


def ann_return(nav: pd.Series) -> float:
    if len(nav) < 2:
        return float("nan")
    return float(nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / len(nav)) - 1.0


def ann_vol(nav: pd.Series) -> float:
    r = _returns(nav)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return float("nan")
    return float((nav / nav.cummax() - 1.0).min())


def sharpe(nav: pd.Series, rf: float = 0.0) -> float:
    vol = ann_vol(nav)
    if not np.isfinite(vol) or vol < 1e-12:
        return float("nan")
    return float((ann_return(nav) - rf) / vol)


def turnover_total(turnover: pd.Series) -> float:
    return float(turnover.sum())


def turnover_ann(turnover: pd.Series, nav: pd.Series) -> float:
    years = len(nav) / TRADING_DAYS
    return float(turnover.sum() / years) if years > 0 else float("nan")


def fee_total(fees: pd.Series) -> float:
    return float(fees.sum())


def rank_ic(factor: pd.DataFrame, panels: dict, rdates, min_names: int = 30) -> pd.Series:
    close, ok = panels["hfq_close"], panels["can_trade"]
    rdates = list(rdates)
    out = {}
    for t, nxt in zip(rdates[:-1], rdates[1:]):
        f = factor.loc[t]
        r = close.loc[nxt] / close.loc[t] - 1.0
        good = ok.loc[t].eq(True) & f.notna() & r.notna()
        if int(good.sum()) < min_names:
            continue
        out[t] = float(f[good].corr(r[good], method="spearman"))
    return pd.Series(out, name="rank_ic", dtype="float64")


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    if n == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan"), "n": 0}
    sd = float(ic.std(ddof=1))
    return {"ic_mean": float(ic.mean()),
            "ic_std": sd,
            "ic_ir": float(ic.mean() / sd * np.sqrt(REBALANCE_PERIODS_PER_YEAR)) if sd else float("nan"),
            "n": n}


def ls_ratio(nav_long: pd.Series, nav_short: pd.Series) -> float:
    return float(nav_long.iloc[-1] / nav_short.iloc[-1]) - 1.0


def ls_diff(nav_long: pd.Series, nav_short: pd.Series) -> float:
    return float(_returns(nav_long).sum() - _returns(nav_short).sum())


def ls_month(nav_long: pd.Series, nav_short: pd.Series, rdates) -> float:
    rl = nav_long.reindex(rdates).pct_change(fill_method=None).dropna()
    rs = nav_short.reindex(rdates).pct_change(fill_method=None).dropna()
    return float((rl - rs).sum())


def ls_summary(nav_df: pd.DataFrame, rdates=None) -> dict:
    assert isinstance(nav_df, pd.DataFrame), \
        f"ls_summary 要的是 5 列净值表(nav_df)，收到的是 {type(nav_df).__name__}（单层净值是一维 Series，别直接传）"
    long_, short_ = nav_df.columns[-1], nav_df.columns[0]
    nl, ns = nav_df[long_], nav_df[short_]
    return {"比值法": ls_ratio(nl, ns),
            "日差法": ls_diff(nl, ns),
            "月差法": ls_month(nl, ns, rdates) if rdates is not None else float("nan")}


def layer_table(nav_df: pd.DataFrame, results=None, rf: float = 0.0) -> pd.DataFrame:
    assert isinstance(nav_df, pd.DataFrame), \
        f"layer_table 要的是 5 列净值表(nav_df)，收到的是 {type(nav_df).__name__}（单层净值是一维 Series，别直接传）"

    rows = []
    for L in nav_df.columns:
        nav = nav_df[L]
        if nav.isna().any():
            raise ValueError(f"层 {L} 的 nav 含 NaN —— 不许静默 dropna")
        row = {"层": L,
               "期末净值": float(nav.iloc[-1]),
               "年化": ann_return(nav),
               "波动": ann_vol(nav),
               "最大回撤": max_drawdown(nav),
               "夏普": sharpe(nav, rf=rf)}
        if results is not None:
            row["累计换手"] = turnover_total(results[L]["turnover"])
            row["费用合计"] = fee_total(results[L]["fees"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("层")


def main():
    import time
    from backtest.panel import load_panels
    from backtest.assemble import build_features, rebalance_dates
    from backtest.factors import FACTORS
    from backtest.weights import weights_for_factor
    from backtest.engine import run_all_layers, nav_frame

    t0 = time.time()
    panels = load_panels()
    rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    res = run_all_layers(W, panels, rdates)
    nav_df = nav_frame(res)

    print(f"面板 {panels['hfq_close'].shape} ｜ 调仓日 {len(rdates)} ｜ 用时 {time.time() - t0:.1f}s")
    print()
    print(layer_table(nav_df, res).round(4).to_string())
    print()
    print("多空（三种定义）：", {k: f"{v:.2%}" for k, v in ls_summary(nav_df, rdates).items()})

    spec = FACTORS["rev_20d"]
    val = spec["fn"](build_features(panels, ["rev_20d"]), {}) * spec["sign"]
    print("IC：", {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in ic_summary(rank_ic(val, panels, rdates)).items()})


if __name__ == "__main__":
    main()
