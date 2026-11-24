import numpy as np
import pandas as pd

from data.cleaner import base
from data.fundamental_merge import TABLES

FIX_CODES = {
    0: "liabilityToAsset 单位 ×100（上游 2024-06 ~ 2025-09 故障）",
}

_TOL = 1e-4              # 实测比值全部 = 0.01000，可以很紧
# 行数少于此【只记账、不断言】：每股只有 27~47 行，3 行异常就是 6.4%，
# 小样本的二项噪声远大于信号。
_MIN_JUDGE_ROWS = 20
# 下面两个阈值是【灾难探测】，不是精密标尺。实测 n>=20 时 fix 最大 0.381、odd 最大 0.086。
# 精确的口径漂移检测在 ops/_fund_health.py —— 那里有全库 1.7 万行，比例才有意义。
_MAX_FIX = 0.60
_MAX_ODD = 0.10
# 年报（statDate 为 12-31）公告滞后下限。年报要审计，短于 30 天不可能。
_MIN_ANNUAL_LAG = 30


def run(df, params, ctx, kind="fundamental") -> pd.DataFrame:
    base.check_not_empty(df, context="fund_quality")
    base.check_required_columns(df, ["assetToEquity", "liabilityToAsset"])
    out = df.copy()
    liab = pd.to_numeric(out["liabilityToAsset"], errors="coerce")
    implied = 1.0 - 1.0 / pd.to_numeric(out["assetToEquity"], errors="coerce")
    ratio = liab / implied

    need = (ratio - 0.01).abs() < _TOL
    out.loc[need, "liabilityToAsset"] = liab[need] * 100.0
    out["fund_fix_bits"] = np.where(need, 1, 0).astype("int64")

    n = int(ratio.notna().sum())
    fix_rate = int(need.sum()) / max(n, 1)
    odd_rate = float((ratio.notna() & ~need & (
        (ratio - 1.0).abs() > 0.02)).sum()) / max(n, 1)

    if n >= _MIN_JUDGE_ROWS:          # 样本太少时比例没有统计意义，只记账不断言
        assert fix_rate < _MAX_FIX, (
            f"liabilityToAsset 单位异常率 {fix_rate:.1%}（{int(need.sum())}/{n}）"
            f" —— 上游口径又变了？")
        assert odd_rate < _MAX_ODD, (
            f"恒等式以未知方式破口 {odd_rate:.1%}（n={n}）"
            f" —— 不是已知的 /100，需要人工看")

    # 年报公告日可疑（滞后 < 30 天）。per-table as-of 会让它变成前视 ——
    # 实测 12 处 / 84533 格 = 0.014%，2 只票。只记录不修复：改不动上游的错误日期。
    short_annual = 0
    if "statDate" in out.columns:
        q4 = out["statDate"].dt.quarter == 4
        for t in TABLES:
            key = f"pubDate_{t}"
            if key not in out.columns:
                continue
            lag = (out[key] - out["statDate"]).dt.days
            short_annual += int((q4 & (lag < _MIN_ANNUAL_LAG)).sum())

    out.attrs["fund_quality"] = {
        "rows": len(out),
        "checked": n,
        "fixed": int(need.sum()),
        "fix_rate": round(fix_rate, 4),
        "short_annual_pubdate": short_annual,
    }
    return out