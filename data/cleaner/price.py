import numpy as np
import pandas as pd
from data.cleaner import base


_PRICE_COLS = ["open", "high", "low", "close"]


def run(df: pd.DataFrame, params: dict, ctx: dict, kind: str = "stock") -> pd.DataFrame:
    base.check_not_empty(df, context="price")
    base.check_required_columns(df)
    base.check_required_columns(df, ["amount"])

    vwap_tolerance = base.as_float(base.read_param(
        params, "vwap_tolerance", 0.02), default=0.02)

    out = df.copy()

    o = pd.to_numeric(out["open"], errors="coerce")
    h = pd.to_numeric(out["high"], errors="coerce")
    lo = pd.to_numeric(out["low"],  errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")
    v = pd.to_numeric(out["volume"], errors="coerce")
    a = pd.to_numeric(out["amount"], errors="coerce")

    price_nan = o.isna() | h.isna() | lo.isna() | c.isna()
    price_nonpos = (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)
    ohlc_violation = ((h < np.maximum(o, c)-1e-6) |
                      (lo > np.minimum(o, c)+1e-6) | (h < lo-1e-6)) & ~price_nan
    out["is_price_bad"] = (price_nan | price_nonpos |
                           ohlc_violation).astype("int8")
    vol_without_amt = (v > 0) & (a == 0)
    amt_without_vol = (a > 0) & (v == 0)
    amt_vol_missing_mismatch = (v.isna() & a.notna() | v.notna() & a.isna())

    can_compute_vwap = (v > 0) & (a > 0) & h.notna() & lo.notna()
    vwap = pd.Series(np.nan, index=out.index)
    vwap[can_compute_vwap] = a[can_compute_vwap]/v[can_compute_vwap]
    lo_bound = lo * (1-vwap_tolerance)
    h_bound = h * (1+vwap_tolerance)
    vwap_outlier = ((vwap < lo_bound) | (vwap > h_bound)).fillna(False)

    out["is_amt_vol_bad"] = (vol_without_amt | amt_without_vol |
                             amt_vol_missing_mismatch | vwap_outlier).astype("int8")
    out.attrs["price"] = {
        "price_nan_rows": int(price_nan.sum()),
        "nonpositive_price_rows": int(price_nonpos.sum()),
        "ohlc_violation_rows": int(ohlc_violation.sum()),
        "price_bad_rows": int(out["is_price_bad"].sum()),
        "vol_pos_amt_zero_rows": int(vol_without_amt.sum()),
        "vol_zero_amt_pos_rows": int(amt_without_vol.sum()),
        "amt_vol_missing_mismatch_rows": int(amt_vol_missing_mismatch.sum()),
        "vwap_outlier_rows": int(vwap_outlier.sum()),
        "amt_vol_bad_rows": int(out["is_amt_vol_bad"].sum()),
        "vwap_tolerance": vwap_tolerance,
    }
    return out
