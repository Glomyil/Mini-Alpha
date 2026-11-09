"""align.py —— 流水线第一格：整备（S1 generic · ③纯清洗）

什么都不筛、不算、不加列；只把数据"摆整齐"，让后面的格子能用：
  日期规范化 → 删坏日期行 → 同日去重(keep first) → 升序排序 → 数值列类型修正
对应事项总表：A3(类型) / A4(日期格式) / A5(去重) / A6(排序)
"""
import pandas as pd

from data.cleaner import base


def run(df: pd.DataFrame, params: dict, ctx: dict) -> pd.DataFrame:
    """入场整备：不改原 df；返回 date 升序唯一、数值列可参与计算的 df"""
    # ① 守卫：空表 / 缺必需列 → 当场报错（别让坏盘子往下流）
    base.check_not_empty(df, context="align")
    base.check_required_columns(df)

    # ② 先复制：不污染调用方手上的原始 df
    out = df.copy()
    n_in = len(out)

    # ③ 日期规范化：解析失败 → NaT → 删掉（align 唯一允许删行的场景）
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    n_bad_date = int(out["date"].isna().sum())      # 数"坏行"：isna，不是 dropna().sum()
    if n_bad_date:
        out = out.dropna(subset=["date"])

    # ④ 先去重(原表里的 first) 再排序(升序)：keep="first" 的语义依赖原始顺序
    out = out.drop_duplicates(subset=["date"], keep="first")
    out = out.sort_values("date")

    # ⑤ 数值列类型修正：只转类型，绝不 fillna（缺值留给下游环节处理）
    # turn（换手率）必须在这里转成数值：抓取层落盘时它是 object（字符串），
    # 不转的话会以 object 一路往下流，某处 to_numeric 后静默变 NaN → 市值全 NaN。
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.reset_index(drop=True)                # 行号清成 0,1,2…

    # ⑥ 记账：把"这一站丢了哪些行"挂在数据的行李牌上（attrs 随 df 传递）
    out.attrs["align"] = {
        "rows_in": n_in,
        "bad_date_rows_dropped": n_bad_date,
        "dup_rows_dropped": (n_in - n_bad_date) - len(out),
        "rows_out": len(out),
    }
    return out
