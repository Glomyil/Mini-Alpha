import pandas as pd
import numpy as np
from data.cleaner import base


def run(df: pd.DataFrame, params: dict, ctx: dict) -> pd.DataFrame:
    base.check_not_empty(df, context="derived")
    base.check_required_columns(df)
    base.check_required_columns(df, ["hfq_close"])

    out = df.copy()

    px = pd.to_numeric(out["hfq_close"], errors="coerce").ffill()
    out["ret_1d_hfq"] = px.pct_change(fill_method=None)
    r = out["ret_1d_hfq"].to_numpy(dtype="float")

    # ── 流通市值（2026-11-04 新增，为中性化准备原料）──
    #   流通股本 = volume / (turn/100)     ← turn 是百分数（实测：sh.600000 反推
    #                                        293.44 亿股 vs 实际 293.52 亿，误差 0.03%）
    #   流通市值 = close(未复权) × 流通股本
    #
    #   ⚠️ 必须用【未复权 close】，不能用 hfq_close —— 复权价是"按今天口径回推的历史价"，
    #      和【当时的股本】相乘不是当日市值。
    #   ⚠️ turn <= 0 或缺失 ⇒ NaN，绝不允许 inf（下游要取 ln，inf 会直接炸）
    #   ⚠️ 没有 turn 列时（老数据 / ETF）不产出该列，让 check_new_columns 保持安静
    mv_stats = {}
    if "turn" in out.columns:
        c = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype="float64")
        v = pd.to_numeric(out["volume"], errors="coerce").to_numpy(dtype="float64")
        t = pd.to_numeric(out["turn"], errors="coerce").to_numpy(dtype="float64")
        shares = np.where(t > 0.0, v / (t / 100.0), np.nan)
        mv = c * shares
        mv[~np.isfinite(mv)] = np.nan          # 除零/溢出统一成 NaN，不留 inf
        mv[mv <= 0.0] = np.nan                 # 市值不可能 <= 0（坏价/坏量）
        out["float_mv"] = mv
        mv_stats = {
            "mv_nan_rows": int(np.isnan(mv).sum()),
            "mv_zero_or_neg_rows": int((c * shares <= 0).sum()) if np.isfinite(c * shares).any() else 0,
        }

    out.attrs["derived"] = {
        "ret_nan_rows": int(np.isnan(r).sum()),
        "ret_inf_rows": int(np.isinf(r).sum()),
        **mv_stats,
    }
    return out
