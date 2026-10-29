from __future__ import annotations
import pandas as pd
import numpy as np
import hashlib
from loguru import logger


def listed(panels):  # 此函数用于检查已有缓存数据是否正常，所需指标是否正常
    ok = panels["hfq_close"].notna()
    return ok.cummax() & ok[::-1].cummax()[::-1]


def _listed_np(panels):
    """和 listed() 一个语义，但走纯 numpy —— 不用 pandas 的 cummax 链（那玩意儿在 1550 万格上开临时数组）。"""
    a = panels["hfq_close"].notna().to_numpy()
    fwd = np.maximum.accumulate(a, axis=0)
    bwd = np.maximum.accumulate(a[::-1], axis=0)[::-1]
    return fwd & bwd


def ret(panels, n):  # 此函数用于校验日收益率变动，即今天收益相较于前一天收益波动的程度
    if n == 1:
        return panels["ret_1d_hfq"].copy()
    # —— 2026-10-13 重写（PITFALLS #23）：不再走 pandas 的 pct_change/shift ——
    # 实测：pandas 的 shift 偶发不写"头部 NaN 块"，那一段保留未初始化内存（看起来像正常数据）；
    # 曾观测到 ret(20) 的预热区漏进 20×5471 个值 → 排名退化 → 分层不变 → 换手塌陷。
    # 这里改成纯 numpy：先造好全 NaN 数组，再显式写 [n:] ⇒ 每个格子都由我们写，结构上不存在"未写区"。
    px = panels["hfq_close"].ffill().to_numpy()          # dtype 跟随面板（float32）
    out = np.full(px.shape, np.nan, dtype=px.dtype)
    out[n:] = px[n:] / px[:-n] - 1.0
    r = pd.DataFrame(out, index=panels["hfq_close"].index,
                     columns=panels["hfq_close"].columns)
    f = pd.DataFrame(np.where(_listed_np(panels), out, np.nan).astype("float32"),
                     index=panels["hfq_close"].index, columns=panels["hfq_close"].columns)
    # 精确不变量：预热 n 行必须全 NaN（同时给 where/astype 也上一道保险）
    _bad = int(f.iloc[:n].notna().to_numpy().sum())
    assert _bad == 0, f"ret({n}) 预热 {n} 行出现 {_bad} 个非 NaN —— 数据退化（PITFALLS #23），拒绝回测"

    logger.info(f"feat ret({n}) fp:{hashlib.md5(f.to_numpy().tobytes()).hexdigest()[:8]} "
                f"nonnan:{int(f.notna().to_numpy().sum())} warmup_bad:{_bad}")

    return f


def _roll(x, n, how):
    """沿时间轴滚动算均值 / 标准差。纯 numpy，内存只要 O(T*N) —— 不开三维临时数组。

    how 只能是 "mean" 或 "std"（std 用 ddof=1）。窗口里只要有一个 NaN，结果就是 NaN
    —— 和 pandas 的 min_periods=n 一致。

    这里用 cumsum 而不是滑窗，是为了把峰值内存压在 O(T*N)。
    老版本每次调用会吃掉大约 2.5 GB（PITFALLS #24），对 2839×5471 的面板是致命的。
    """
    x = np.asarray(x, dtype="float64")
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype="float64")
    if T < n:
        return out

    def wsum(a):
        c = np.cumsum(a, axis=0)
        c = np.vstack([np.zeros((1, a.shape[1])), c])
        return c[n:] - c[:-n]

    # inf 必须当"无效值"处理：amount == 0 时这里会除零得到 inf，
    # 而 inf 一旦进了 cumsum，后面所有窗口都会被毒掉（inf - inf = NaN ✗）。
    # 下游一律用 isfinite 过滤，inf 和 NaN 等价 ⇒ 和原来 pandas 版结果一致 ✓
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    s1 = wsum(v)
    valid = wsum(ok.astype("float64")) == n
    if how == "mean":
        res = s1 / n
    else:
        s2 = wsum(v * v)
        var = (s2 - s1 * s1 / n) / (n - 1)
        res = np.sqrt(np.where(var < 0, 0.0, var))
    out[n - 1:] = np.where(valid, res, np.nan)
    return out


def _warm(fn, f, n):
    """检查 warm-up 期、打指纹 —— 和 ret() 同一套纪律。

    rolling(n) 的窗口要到第 n-1 行才凑得满，所以 warm-up 区是【前 n-1 行】必须是 NaN。
    注意 ret() 的构造不一样：它的前 n 行都是 NaN。差一行，别照着抄错。
    """
    warm = n - 1
    arr = f.to_numpy()
    _bad = int(f.iloc[:warm].notna().to_numpy().sum())
    assert _bad == 0, (f"{fn.__name__}({n}) warm-up: {_bad} non-NaN in first {warm} rows "
                       f"-- data degraded (PITFALLS #23), refusing to continue")
    logger.info(f"feat {fn.__name__}({n}) fp:{hashlib.md5(arr.tobytes()).hexdigest()[:8]} "
                f"nonnan:{int(np.count_nonzero(~np.isnan(arr)))} warmup_bad:{_bad}")


def vol(panels, n):
    px = panels["ret_1d_hfq"]
    out = _roll(px.to_numpy(), n, "std")
    f = pd.DataFrame(out.astype("float32"), index=px.index, columns=px.columns)
    _warm(vol, f, n)
    return f


def ma_dev(panels, n):
    px = panels["hfq_close"].ffill()
    num = px.to_numpy(dtype="float64")
    m = _roll(num, n, "mean")
    out = np.full(num.shape, np.nan, dtype="float64")
    np.divide(num, m, out=out)
    out -= 1.0
    f = pd.DataFrame(out.astype("float32"), index=px.index, columns=px.columns)
    _arr = np.where(_listed_np(panels), out, np.nan)
    f = pd.DataFrame(_arr.astype("float32"), index=px.index, columns=px.columns)
    _warm(ma_dev, f, n)
    return f


def amihud(panels, n):
    px = panels["ret_1d_hfq"]
    x = (px.abs() / panels["amount"]).to_numpy()
    out = _roll(x, n, "mean")
    f = pd.DataFrame(out.astype("float32"), index=px.index, columns=px.columns)
    _warm(amihud, f, n)
    return f


FEATURES = {"ret": ret, "vol": vol, "ma_dev": ma_dev, "amihud": amihud}

if __name__ == "__main__":
    # 冒烟：最小假面板（4 天 × 3 只），四张表用**真键名**，停牌 / 干净 / 晚上市 三种形态都放进去
    days = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    close = pd.DataFrame({"sh.600000": [10.0, np.nan, 11.0, 12.1],     # 停牌缺行
                          "sz.000001": [10.0, 20.0, 40.0, 80.0],      # 干净
                          "sz.300001": [np.nan, np.nan, 5.0, 5.0]},   # 晚上市
                         index=days)
    ok = close.notna().cummax() & close.notna()[::-1].cummax()[::-1]
    panels = {"hfq_close": close.astype("float32"),
              "hfq_open": close.astype("float32"),
              "ret_1d_hfq": close.ffill().pct_change(fill_method=None).where(ok).astype("float32"),
              "can_trade": close.notna().astype(bool)}
    for name in ("ret", "vol", "ma_dev"):
        out = FEATURES[name](panels, n=2)
        print(
            f"  {name}(2): {out.shape} {out.dtypes.iloc[0]}  非 NaN {int(out.notna().to_numpy().sum())} 格")
    print("冒烟通过 ✅  完整验收：python tests/test_features.py")