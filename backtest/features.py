from __future__ import annotations
import pandas as pd
import numpy as np
import hashlib
from loguru import logger


def listed(panels):  # 此函数用于检查已有缓存数据是否正常，所需指标是否正常
    ok = panels["hfq_close"].notna()
    return ok.cummax() & ok[::-1].cummax()[::-1]


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
    f = r.where(listed(panels)).astype("float32")
    # 精确不变量：预热 n 行必须全 NaN（同时给 where/astype 也上一道保险）
    _bad = int(f.iloc[:n].notna().to_numpy().sum())
    assert _bad == 0, f"ret({n}) 预热 {n} 行出现 {_bad} 个非 NaN —— 数据退化（PITFALLS #23），拒绝回测"

    logger.info(f"feat ret({n}) fp:{hashlib.md5(f.to_numpy().tobytes()).hexdigest()[:8]} "
                f"nonnan:{int(f.notna().to_numpy().sum())} warmup_bad:{_bad}")

    return f


def vol(panels, n):  # 计算波动率，即用一天收益率变化率用标准差公式计算
    return panels["ret_1d_hfq"].rolling(n, min_periods=n).std().astype("float32")


def ma_dev(panels, n):  # 计算日收益偏差均值的程度，即用日收益除以日收益均值减一的结果
    f = panels["hfq_close"].ffill()
    return (f/f.rolling(n, min_periods=n).mean()-1).where(listed(panels)).astype("float32")


def amihud(panels, n):  # 计算流动性指标，即用一天收益率变化除以一天成交量，然后求出平均值，平均值越大代表单次成交对整体价格的影响大，即流动性低
    return (panels["ret_1d_hfq"].abs()/panels["amount"]).rolling(n, min_periods=n).mean().astype("float32")


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
