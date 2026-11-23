"""align.py —— 流水线第一格：整备（S1 generic · ③纯清洗）

什么都不筛、不算、不加列；只把数据"摆整齐"，让后面的格子能用：
  日期规范化 → 删坏日期行 → 同日去重(keep first) → 升序排序 → 数值列类型修正
对应事项总表：A3(类型) / A4(日期格式) / A5(去重) / A6(排序)
"""
import pandas as pd

from data.cleaner import base


# 时间列名按数据源取。加第三类数据源时在这里加一行。
_TIME_COL = {"stock": "date", "fundamental": "statDate"}


def run(df: pd.DataFrame, params: dict, ctx: dict,
        kind: str = "stock") -> pd.DataFrame:
    """入场整备：不改原 df；返回时间列升序唯一、数值列可参与计算的 df。

    kind 不传 = 行情，现有调用零改动；财务传 kind="fundamental"。
    它同时决定两件事：① 必需列契约（交给 base 分派）② 时间列叫什么。

    数值列仍是行情那 7 个 —— 财务表里一个都不存在，下面 for 循环的
    `if col in out.columns` 会让它天然空转（32 个指标 merge 时已 TRY_CAST 成 DOUBLE）。
    """
    if kind not in _TIME_COL:
        raise ValueError(f"未知的数据类型 kind={kind!r}，可选 {sorted(_TIME_COL)}")
    tcol = _TIME_COL[kind]

    # ① 守卫：空表 / 缺必需列 → 当场报错（别让坏盘子往下流）
    base.check_not_empty(df, context="align")
    base.check_required_columns(df, kind=kind)

    # ② 先复制：不污染调用方手上的原始 df
    out = df.copy()
    n_in = len(out)

    # ③ 时间列规范化：解析失败 → NaT → 删掉（align 唯一允许删行的场景）
    out[tcol] = pd.to_datetime(out[tcol], errors="coerce")
    n_bad_date = int(out[tcol].isna().sum())        # 数"坏行"：isna，不是 dropna().sum()
    if n_bad_date:
        out = out.dropna(subset=[tcol])

    # ④ 先去重(原表里的 first) 再排序(升序)：keep="first" 的语义依赖原始顺序
    out = out.drop_duplicates(subset=[tcol], keep="first")
    out = out.sort_values(tcol)

    # ⑤ 数值列类型修正：只转类型，绝不 fillna（缺值留给下游环节处理）
    # turn（换手率）必须在这里转成数值：抓取层落盘时它是 object（字符串），
    # 不转的话会以 object 一路往下流，某处 to_numeric 后静默变 NaN → 市值全 NaN。
    # 估值四列同样是"抓取层落盘为字符串"的字段，必须在这里转数值 ——
    # 否则会以 object 一路往下流，某处 to_numeric 后静默变 NaN（turn 踩过的同一个坑）。
    for col in ["open", "high", "low", "close", "volume", "amount", "turn",
                "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.reset_index(drop=True)                # 行号清成 0,1,2…

    # ⑥ 记账：把"这一站丢了哪些行"挂在数据的行李牌上（attrs 随 df 传递）
    out.attrs["align"] = {
        "kind": kind,
        "time_col": tcol,
        "rows_in": n_in,
        "bad_date_rows_dropped": n_bad_date,
        "dup_rows_dropped": (n_in - n_bad_date) - len(out),
        "rows_out": len(out),
    }
    return out
