import pandas as pd
from data.cleaner import base


def run(df: pd.DataFrame, params: dict, ctx: dict, kind: str = "stock") -> pd.DataFrame:  # 所有模块统一契约接口名称用于data的处理
    base.check_not_empty(df, context="adjust")  # 检查数据复权因子是否存在
    base.check_required_columns(df)
    base.check_required_columns(
        df, ["back_adj_factor"])  # 检查白名单中是否有出现流程以外的数据类型

    out = df.copy()
    f = pd.to_numeric(out["back_adj_factor"],
                      errors="coerce")  # 将复权因子转换为浮点数用于计算
    # baostock数据源采用的是比例复权，所以实际上复权价格等于收盘价×复权因子
    out["hfq_close"] = pd.to_numeric(out["close"], errors="coerce")*f
    out["hfq_open"] = pd.to_numeric(out["open"], errors="coerce")*f

    out.attrs["adjust"] = {  # 产生这个环节的数据报告对应["adjust"]环节
        "hfq_nan_rows": int(out["hfq_close"].isna().sum()),
        "hfq_open_nan_rows": int(out["hfq_open"].isna().sum()),
        "factor_nonpositive_rows": int((f <= 0).sum()),
    }

    return out
