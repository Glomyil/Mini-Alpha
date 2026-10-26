import pandas as pd
import numpy as np

from data.cleaner import base

CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")   # 创业板注册制改革施行日
MAIN_5D_NEW_FROM = pd.Timestamp("2023-04-10")  # 主板全面注册制首批上市日

CENT = 100.0


def board_of(code: str) -> str:
    """从股票代码判板块。未知前缀直接抛错"""
    c = str(code).split(".")[-1]
    if c[:3] in ("300", "301", "302"):   # 302 = 创业板新增号段（2026-10-23 数据反推确认 20%）
        return "chinext"
    if c[:3] in ("688", "689"):
        return "star"
    if c[:1] in ("8", "4"):
        return "bse"
    if c[:2] in ("60", "00"):
        return "main"
    raise ValueError(f"未知板块前缀：code={code!r}——请先在 board_of 里登记它的涨跌幅规则")


def compute_limit_pct(board: str, dates, is_st) -> np.ndarray:
    """当日适用的涨跌幅上限。ST 只对主板生效（5%）。"""
    n = len(dates)
    st = np.asarray(is_st).astype(bool)
    if board == "main":
        return np.where(st, 0.05, 0.10).astype("float64")
    if board == "chinext":
        d = pd.to_datetime(pd.Series(dates)).to_numpy()
        old = d < np.datetime64(CHINEXT_20PCT_FROM)
        return np.where(old, 0.10, 0.20).astype("float64")
    if board == "star":
        return np.full(n, 0.20, dtype="float64")
    if board == "bse":
        return np.full(n, 0.30, dtype="float64")
    raise ValueError(f"未知板块：{board!r}")


def compute_new_listing_days(board: str, dates) -> np.ndarray:
    """新股期长度（交易日）。主板 2023-04-10 前只有上市首日，之后改为前 5 日。"""
    n = len(dates)
    if board != "main":
        return np.full(n, 5, dtype="int16")
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    return np.where(d < np.datetime64(MAIN_5D_NEW_FROM), 1, 5).astype("int16")


def _round_cent(x: np.ndarray) -> np.ndarray:
    """根据交易所的规则四舍五入到分—— 不能用 Python 的 round 。"""
    return np.floor(np.asarray(x, dtype="float64") * CENT + 0.5) / CENT


def compute_limits(prev_close, open_px, pct, days_since_listing,
                   new_listing_days, factor_changed, tradeable, resumed=None):
    """纯函数：算限价 + 判 flag。所有入参都是等长数组。

    resumed：长期停牌后【复牌首日】——A 股复牌首日不设涨跌幅限制，
             所以这天不判涨跌停（实测证据：sz.302132 在 2016-05-25 跌 34.20%）。
             None 表示没有这一天（向后兼容）。
    """
    prev = np.asarray(prev_close, dtype="float64")
    opn = np.asarray(open_px, dtype="float64")
    p = np.asarray(pct, dtype="float64")
    if resumed is None:
        resumed = np.zeros(prev.shape[0], dtype=bool)

    limit_up = _round_cent(prev * (1.0 + p))
    limit_down = _round_cent(prev * (1.0 - p))

    ok = (np.asarray(tradeable).astype(bool)
          & ~np.asarray(factor_changed).astype(bool)
          & ~np.asarray(resumed).astype(bool)
          & ~(np.asarray(days_since_listing) <= np.asarray(new_listing_days)).astype(bool))

    flag_up = (np.abs(opn - limit_up) <= 0.005) & ok
    flag_down = (np.abs(opn - limit_down) <= 0.005) & ok
    return {
        "limit_up": limit_up,
        "limit_down": limit_down,
        "flag_up": flag_up.astype("int8"),
        "flag_down": flag_down.astype("int8"),
    }


def run(df: pd.DataFrame, params: dict, ctx: dict) -> pd.DataFrame:
    base.check_not_empty(df, context="limits")
    base.check_required_columns(df)
    base.check_required_columns(
        df, ["close", "open", "back_adj_factor", "isST", "tradestatus"])

    out = df.copy()
    board = board_of(ctx["code"])
    dates = out["date"]

    close = pd.to_numeric(out["close"], errors="coerce").astype("float64")
    open_px = pd.to_numeric(out["open"], errors="coerce").astype("float64")
    prev_close = close.shift(1)

    fac = pd.to_numeric(out["back_adj_factor"],
                        errors="coerce").astype("float64")
    fac_prev = fac.shift(1)
    factor_changed = (fac.sub(fac_prev).abs() >
                      1e-9).to_numpy() & fac_prev.notna().to_numpy()

    # 复牌首日：昨天停牌、今天恢复交易 —— 那天不设涨跌幅限制（不判涨跌停）
    was_suspended = out["tradestatus"].shift(1).eq(0)
    resumed = (was_suspended & out["tradestatus"].eq(1)).to_numpy()

    days_since = np.arange(1, len(out) + 1, dtype="int16")     # 首行 = 第 1 天
    pct = compute_limit_pct(board, dates, out["isST"])
    n_days = compute_new_listing_days(board, dates)

    res = compute_limits(
        prev_close=prev_close.to_numpy(),
        open_px=open_px.to_numpy(),
        pct=pct,
        days_since_listing=days_since,
        new_listing_days=n_days,
        factor_changed=factor_changed,
        tradeable=(out["tradestatus"] == 1).to_numpy(),
        resumed=resumed,
    )

    out["flag_open_limit_up"] = res["flag_up"]
    out["flag_open_limit_down"] = res["flag_down"]
    out["flag_new_listing"] = (days_since <= n_days).astype("int8")

    out.attrs["limits"] = {
        "n_limit_up": int(res["flag_up"].sum()),
        "n_limit_down": int(res["flag_down"].sum()),
        "n_new_listing": int(out["flag_new_listing"].sum()),
        "n_factor_changed": int(factor_changed.sum()),
        "n_missing_prev": int(prev_close.isna().sum()),
        "n_resumed": int(resumed.sum()),
    }
    return out