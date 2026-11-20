import pandas as pd
from data.cleaner import base

_PRICE_COLS = ["open", "high", "low", "close"]


def run(df: pd.DataFrame, params: dict, ctx: dict, kind: str = "stock") -> pd.DataFrame:  # 统一数据接口契约，怎么进来怎么出去
    base.check_not_empty(df, context="suspension")
    base.check_required_columns(df)
    base.check_required_columns(df, ["tradestatus"])  # 检查是否有缺列，是否有环节所需的列

    out = df.copy()
    # 将tradestatus变为可以衡量的数字，缺少会填充为nan
    ts = pd.to_numeric(out["tradestatus"], errors="coerce")
    susp = ts == 0  # 将ts的结果与0对比，用布尔值的形式处理呈现结果
    # 将统计出来的结果用is_suspended列存储，添加进data frame
    out["is_suspended"] = susp.astype("int8")

    grp = (~susp).cumsum()  # 利用向量化游程计算实际连续停牌的天数
    out["suspension_days"] = susp.groupby(grp).cumsum().astype(
        "int16")  # 划分为不同的组，将计算出来的结果新开一个单独的列存储停牌天数
    seg_len = susp.groupby(grp).transform("sum")
    out["is_resume"] = (susp.shift(1, fill_value=False)  # 计算并添加一列标注复牌日的数据
                        & ~susp).astype("int8")
    out["resume_after_days"] = (seg_len.shift(  # 这里新添加一列标注停牌长度的数据，用于因子计算
        1).where(out["is_resume"] == 1, 0).fillna(0).astype("int16"))

    vol = pd.to_numeric(out["volume"], errors="coerce")
    has_price = out[_PRICE_COLS].notna().all(axis=1)  # 统计四个价格都存在的数据
    all_price_na = out[_PRICE_COLS].isna().all(axis=1)  # 统计四个价格都缺失的数据

    out.attrs["suspension"] = {
        "suspended_rows": int(susp.sum()),  # 统计停牌的行数
        "zombie_rows": int((susp & has_price & (vol == 0)).sum()),  # 僵尸行异常数据统计
        # 空量行异常数据统计
        "nan_volume_rows": int((susp & has_price & vol.isna()).sum()),
        "empty_rows": int((susp & all_price_na).sum()),  # 全空行异常数据统计
        # 统计停牌，但是却有交易的异常数据
        "contradiction_rows": int((susp & (vol > 0)).sum()),
    }
    return out
