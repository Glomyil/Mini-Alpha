from __future__ import annotations
import pandas as pd
import numpy as np
KNOWN_EXPOSURES = frozenset({"industry", "ln_float_mv"})


def _cross_sectional_rank(df: pd.DataFrame) -> np.ndarray:
    # 逐期【横截面】百分位秩（0~1）。不是时序秩 ——
    # 时序秩（同一只票在历史上排名）会毁掉横截面可比性 ⇒ 分层直接用不了。
    return df.rank(axis=1, pct=True, na_option="keep").to_numpy(dtype="float64")


def neutralize(factor: pd.DataFrame, panels: dict, rdates, *, exposures=("industry", "ln_float_mv"),
               mv_floor: float = 0.0, weight=None, drop_first: bool = True, min_n: int = 10):

    exposures = tuple(exposures)
    if not exposures:
        # 开中性化却一项都不剥 = 只减了横截面均值。那是有效操作，但【不是中性化】，
        # 允许它等于给"以为做了中性化其实没做"留后门 ⇒ 直接拒。要"不剥"用 enabled: false。
        raise ValueError(
            "exposures 不能为空：开中性化却一项都不剥只等于减去横截面均值；"
            "要「不剥」请把 neutralize.enabled 设为 false")
    bad = [e for e in exposures if e not in KNOWN_EXPOSURES]
    if bad:
        raise ValueError(f"未知暴露项 {bad}；可用：{sorted(KNOWN_EXPOSURES)}")
    # 只有真要剥行业/市值时才需要对应的原料表 —— 别因为缺一张不相干的表把整件事拦掉
    _need = {"industry": "industry", "ln_float_mv": "float_mv"}
    _miss = [_need[e] for e in exposures if _need[e] not in panels]
    if _miss:
        raise KeyError(f"panels 缺 {_miss}（T-6.1 的原料表）")

    ind_all = panels["industry"].to_numpy()
    mv_all = panels["float_mv"].to_numpy(dtype="float64")
    rk = _cross_sectional_rank(factor)
    out = np.full(rk.shape, np.nan)      # 没算的就是 NaN，绝不用 0 冒充

    for t in rdates:
        if t not in factor.index:
            continue
        i = factor.index.get_loc(t)
        y, mv, ind = rk[i], mv_all[i], ind_all[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            lnmv = np.log(mv)
        # 只有在【要剥市值】时才要求 ln(市值) 有效 —— 否则缺市值的票会被无谓地剔掉
        ok = np.isfinite(y)
        if "ln_float_mv" in exposures:
            ok = ok & np.isfinite(lnmv) & (mv > mv_floor)
        if ok.sum() < min_n:             # 样本太少 ⇒ 这一期不算（留 NaN）
            continue
        y, lnmv, ind = y[ok], lnmv[ok], ind[ok]

        # 设计矩阵 = [截距] + 按 exposures 逐项加列
        #   ⚠️ 必须真的按 exposures 分支：只校验不执行的话，yml 里写 exposures: [industry]
        #      实际仍会剥市值 —— 而 banner 照样印「暴露:industry」⇒ 口径撒谎（2026-11-10 修的）
        cols = [np.ones(len(y))]
        if "industry" in exposures:
            # 不 drop 一列 ⇒ 哑变量之和 = 截距 ⇒ 奇异。
            # drop_first 就一定要去掉一列：某期只剩 1 个行业时也必须去
            #（那时 ids[1:] 是空 ⇒ 只剩截距）—— 留着会造一个全 1 的哑变量 ≡ 截距 ⇒ 秩亏
            ids = np.unique(ind)
            use = ids[1:] if drop_first else ids
            cols += [(ind == k).astype("float64") for k in use]
        if "ln_float_mv" in exposures:
            cols.append(lnmv)
        X = np.column_stack(cols)

        if weight == "float_mv":          # WLS：行乘 sqrt(w) 等价于加权最小二乘
            w = np.sqrt(mv[ok])
            Xw = X * w[:, None]
            beta, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
            r = (y * w - Xw @ beta) / w
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta

        out[i, np.nonzero(ok)[0]] = r

    return pd.DataFrame(out, index=factor.index, columns=factor.columns)
