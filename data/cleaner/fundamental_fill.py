import pandas as pd
import numpy as np

from data.cleaner import base
from data.fundamental_merge import TABLES


def run(df, params, ctx, kind="fundamental") -> pd.DataFrame:
    base.check_not_empty(df, context="fundamental_fill")
    base.check_required_columns(df, ["statDate"])
    if "trading_days" not in ctx:
        raise KeyError("ctx 缺 'trading_days'（交易日索引）"
                       " —— fund_fill 需要它把期频摊到日频")
    # 排序前置：让 out 与 left 同序，写回不依赖 label 对齐（隐式正确要消掉）
    days = pd.DatetimeIndex(ctx["trading_days"]).sort_values()
    left = pd.DataFrame({"date": days}).sort_values("date")   # merge_asof 要求升序
    out = pd.DataFrame(index=pd.Index(days, name="date"))
    out["statDate"] = pd.NaT
    for table, cols in TABLES.items():
        key = f"pubDate_{table}"
        base.check_required_columns(df, [key] + cols)
        sub = df.loc[df[key].notna(), ["statDate", key] + cols]
        #dtype 必须与 left 一致：事件表是 [us]，交易日是 [ns]
        sub = sub.assign(**{key: sub[key].astype("datetime64[ns]"),
                            "statDate": sub["statDate"].astype("datetime64[ns]")})
        if not sub.empty:
            # 同一 pubDate 的多期（年报 + 一季报同日公告，实测 6.9% 的行）构成该日的
            # 一次完整披露：每列取当天最后一次出现的【非空】值。
            # GroupBy.last() 逐列跳过 NaN，所以「只有年报披露的字段」能存活，
            # 必须先按 statDate 升序：last() 取的是组内【当前行序】的最后一条。
            sub = (sub.sort_values(["statDate"])
                      .groupby(key, as_index=False)
                      .last()
                      .sort_values(key))
        if sub.empty:
            for c in cols:
                out[c] = np.nan
            out[key] = pd.NaT
            out[f"lag_{table}"] = np.nan
            continue

        m = pd.merge_asof(left, sub,
                          left_on="date", right_on=key,
                          direction="backward",
                          allow_exact_matches=True)   # 同日公告当天可见（成交在 T+1 开盘）
        m = m.set_index("date")

        out[cols] = m[cols]                              # ← 不填充，NaN 就是 NaN
        out[key] = m[key]                                # 保留本表的公告日（追溯/审计用）
        out[f"lag_{table}"] = (m.index.to_series(index=m.index) - m[key]).dt.days

        if table == "profit":
            out["statDate"] = m["statDate"]

    # ── 可查性 ──
    # 为什么要记：NaN 有两种来源 ——「这期没公告」和「我们没下到」。
    # 摊到日频后两者长得一模一样，所以必须在源头按表计数。
    src_periods = len(df)
    na_pub = {t: int(df[f"pubDate_{t}"].isna().sum()) for t in TABLES}
    ind_all = [c for cols in TABLES.values() for c in cols]
    out.attrs["fundamental_fill"] = {
        "trading_days": len(days),
        "source_periods": src_periods,
        # 各表「pubDate 为 NaT」的期数：= 该表在这些期没公告（或没下到）
        "periods_without_pubdate": na_pub,
        # 整张表全空 ⇒ 大概率是没下到，不是没公告
        "tables_entirely_missing": sorted(
            t for t, n in na_pub.items() if src_periods > 0 and n == src_periods),
        # 日频里 32 个指标全空的行数：首份公告之前 + 完全无数据
        "rows_with_no_fundamental": int(out[ind_all].isna().all(axis=1).sum()),
    }

    return out
