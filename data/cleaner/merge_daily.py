import numpy as np
import pandas as pd

from data.cleaner import base
from data.cleaner import fundamental_fill


def merge(price_df: pd.DataFrame, fund_df: pd.DataFrame | None) -> pd.DataFrame:
    base.check_not_empty(price_df, context="merge_daily")
    base.check_required_columns(price_df, ["date"])
    assert price_df["date"].is_monotonic_increasing, (
        "price_df 的 date 不是升序 —— 上游 align 出问题了？")

    cols = fundamental_fill.output_columns()
    out = price_df.copy()
    n_rows = len(out)

    if fund_df is None:
        for c in cols:
            out[c] = pd.NaT if (
                c == "statDate" or c.startswith("pubDate_")) else np.nan
    else:
        missing = [c for c in cols if c not in fund_df.columns]
        assert not missing, (
            f"财务表缺列 {missing[:5]}… —— fundamental_fill 的产出契约变了？")

        out = out.merge(fund_df[cols], left_on="date", right_index=True,
                        how="left", validate="one_to_one")

    assert len(out) == n_rows, (
        f"行数变了：{n_rows} → {len(out)} —— merge 绝不能缩样本")
    assert list(out.columns) == list(price_df.columns) + cols, (
        f"列不合契约：期望 {len(price_df.columns) + len(cols)} 列，实际 {len(out.columns)} 列")

    fin = out[cols]
    out.attrs["merge_daily"] = {
        "fund_present": fund_df is not None,
        "rows": n_rows,
        "cols": len(out.columns),
        # 有几行真的带财务值（首份公告之前的行必然是 0）
        "rows_with_fundamental": int(fin.notna().any(axis=1).sum()),
        # 哪几列在这票上全空 —— 全库汇总后，这就是 panel.FIELDS 选列的依据
        "fund_cols_all_nan": [c for c in cols if not fin[c].notna().any()],
    }
    return out
