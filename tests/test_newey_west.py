"""tests/test_newey_west.py —— Newey-West（HAC）标准误的回归网

为什么要它：现在所有 t 值都假设「每期独立」。但策略收益和 IC 都有自相关
（行业轮动、风格切换都有惯性）⇒ 朴素 t 被系统性高估。

三个层次的验证，期望值来源各不相同：
  ① 手算：T=4、L=1 的极小样本，SE=0.625 是纸上推的，不是跑出来的
  ② 结构恒等式：L=0 时必须【逐位等于】朴素 SE（NW 退化成普通标准误）
  ③ 独立实现交叉验证：与 statsmodels 的 cov_hac(use_correction=False) 对拍

不测：lag 选择的最优性（那是研究选择，不是实现正确性）
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.metrics import newey_west_se, nw_lags, t_stat, t_stat_nw  # noqa: E402

CK = []


def ck(cond, msg):
    CK.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


def p_from_t(t):
    """双尾 p，和 metrics.t_stat 里的公式同源（正态近似）"""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


# ============================================================================
# ① 手算：x = [1,2,3,4]，L=1
#    均值 2.5，离差 e = [-1.5, -0.5, 0.5, 1.5]
#    g0 = (2.25+0.25+0.25+2.25)/4 = 1.25
#    g1 = [(-0.5)(-1.5) + (0.5)(-0.5) + (1.5)(0.5)] / 4 = 1.25/4 = 0.3125
#    Bartlett 权重 (1 - 1/(L+1)) = 0.5
#    S = 1.25 + 2*0.5*0.3125 = 1.5625
#    未修正 SE = sqrt(S/T) = sqrt(1.5625/4) = 0.625
#    小样本修正 T/(T-1) = 4/3 ⇒ SE = 0.625 * sqrt(4/3) = 0.7216878365...  <- 纸上推的
#    （这条修正不是可选的：不乘的话 L=0 就对不上报告里那列朴素 SE）
# ============================================================================
print("\n\u2500\u2500 \u2460 \u624b\u7b97\u6781\u5c0f\u6837\u672c\u2500\u2500")
x = np.array([1.0, 2.0, 3.0, 4.0])
se1 = newey_west_se(x, lags=1)
want = 0.625 * math.sqrt(4.0 / 3.0)
ck(abs(se1 - want) < 1e-12, f"T=4 L=1 \u21d2 SE = {se1:.12f}\uff08\u624b\u7b97 {want:.12f}\uff09")

naive = float(x.std(ddof=1) / np.sqrt(len(x)))
se0 = newey_west_se(x, lags=0)
ck(abs(se0 - naive) < 1e-12, f"L=0 \u21d2 \u9000\u5316\u6210\u666e\u901a\u6807\u51c6\u8bef {naive:.12f}")


# ============================================================================
# ② 结构恒等式：L=0 必须在任意序列上都等于朴素 SE
# ============================================================================
print("\n\u2500\u2500 \u2461 L=0 \u5fc5\u987b\u7b49\u4e8e\u6734\u7d20 SE\u2500\u2500")
rng = np.random.default_rng(0)
for n in (10, 131, 552):
    s = rng.normal(0.002, 0.05, n)
    a = newey_west_se(s, lags=0)
    b = float(s.std(ddof=1) / np.sqrt(n))
    ck(abs(a - b) < 1e-12, f"n={n} \u21d2 {a:.10f} == {b:.10f}")


# ============================================================================
# ③ 与 statsmodels.cov_hac 对拍（独立实现，不是同一套代码）
#    cov_hac 默认 use_correction=True（乘 T/(T-1)）⇒ 我们的默认也开着，才能对得上
# ============================================================================
print("\n\u2500\u2500 \u2462 \u4e0e statsmodels \u5bf9\u62cd\u2500\u2500")
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac

    def _sm_se(s, L):
        X = np.ones((len(s), 1))
        res = sm.OLS(s, X).fit()
        V = cov_hac(res, nlags=L, use_correction=True)   # 默认 weights_func=bartlett
        return float(np.sqrt(V[0, 0]))

    n, phi = 300, 0.6
    e = rng.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    for L in (0, 1, 4, 8, 12):
        mine = newey_west_se(y, lags=L)
        theirs = _sm_se(y, L)
        ck(abs(mine - theirs) < 1e-9, f"L={L:2d} \u21d2 \u672c\u5b9e\u73b0 {mine:.10f} vs statsmodels {theirs:.10f}")
except ImportError:
    print("  \u26a0 statsmodels \u672a\u5b89\u88c5\uff0c\u8df3\u8fc7\u4ea4\u53c9\u9a8c\u8bc1")


# ============================================================================
# ④ 方向性：正自相关 ⇒ NW 变大；负自相关 ⇒ NW 变小（抓符号写反）
# ============================================================================
print("\n\u2500\u2500 \u2463 \u81ea\u76f8\u5173\u65b9\u5411\u2500\u2500")
n = 400
e = rng.normal(0, 1, n)
pos = np.zeros(n)
neg = np.zeros(n)
for i in range(1, n):
    pos[i] = 0.7 * pos[i - 1] + e[i]
    neg[i] = -0.7 * neg[i - 1] + e[i]
L = nw_lags(n)
ck(newey_west_se(pos, L) > float(pos.std(ddof=1) / np.sqrt(n)) * 1.5,
   f"\u6b63\u81ea\u76f8\u5173 \u21d2 NW SE \u660e\u663e\u53d8\u5927\uff08L={L}\uff09")
ck(newey_west_se(neg, L) < float(neg.std(ddof=1) / np.sqrt(n)) * 0.8,
   f"\u8d1f\u81ea\u76f8\u5173 \u21d2 NW SE \u53d8\u5c0f\uff08L={L}\uff09")


# ============================================================================
# ⑤ t_stat_nw：L=0 时与现有 t_stat 完全一致（证明只是加了 NW，没动别的）
# ============================================================================
print("\n\u2500\u2500 \u2464 t_stat_nw \u4e0e t_stat \u7684\u4e00\u81f4\u6027\u2500\u2500")
# ⚠️ 不能拿随机正态序列来断言「NW |t| 更小」—— 它自相关接近 0，
#    NW 可能因为小样本噪声变大也可能变小。必须造【确定有正自相关】的序列。
s = rng.normal(0.001, 0.03, 130)
t0, p0 = t_stat(s)
t1, p1 = t_stat_nw(s, lags=0)
ck(abs(t0 - t1) < 1e-12 and abs(p0 - p1) < 1e-12, f"L=0 \u21d2 t={t1:.10f} == {t0:.10f}")

ar = np.zeros(130)
ar[0] = s[0]
for i in range(1, 130):
    ar[i] = 0.6 * ar[i - 1] + s[i]
ta, _ = t_stat(ar)
tb, pb = t_stat_nw(ar, lags=nw_lags(len(ar)))
ck(abs(pb - p_from_t(tb)) < 1e-12, f"NW t={tb:.4f} \u21d2 p={pb:.6f}\uff08p \u5fc5\u987b\u7531\u65b0 t \u91cd\u7b97\uff09")
ck(abs(tb) < abs(ta),
   f"\u786e\u5b9a\u6b63\u81ea\u76f8\u5173(AR1 phi=0.6) \u21d2 NW |t| \u5fc5\u987b\u5c0f\u4e8e\u6734\u7d20\uff08{tb:.4f} vs {ta:.4f}\uff09")



# ============================================================================
# ⑥ ic_summary 的 NW 版 IC-IR：恒等式 IC-IR = t * sqrt(ppy/n)
#    所以 lags=0 时、NW 版必须【逐位等于】朴素版（不是近似）。
#    这条抓的是「接线错了」：把 ppy/n 写反、或忘了 sqrt。
# ============================================================================
print("\n── ⑥ ic_summary 的 NW 版 IC-IR──")
import pandas as pd  # noqa: E402
from backtest.metrics import ic_summary  # noqa: E402

_idx = pd.date_range("2016-01-31", periods=129, freq="ME")
_ic = pd.Series(rng.normal(0.03, 0.15, 129), index=_idx)
_a = ic_summary(_ic, periods_per_year=12)
_b = ic_summary(_ic, periods_per_year=12, lags=0)
# ⚠️ 必须用 _b（lags=0）去比，不能拿 _a（经验法则 lag=4）比 —— 那是两个不同的 lag。
ck(abs(_b["ic_ir_nw"] - _b["ic_ir"]) < 1e-12,
   f"lags=0 ⇒ NW 版 {_b['ic_ir_nw']:.10f} == 朴素版 {_b['ic_ir']:.10f}")
ck(abs(_a["ic_ir"] - _b["ic_ir"]) < 1e-12, "朴素 IC-IR 不受 lags 影响")
# lags 不同 ⇒ NW 版变化（否则说明参数没传进去）
_c = ic_summary(_ic, periods_per_year=12, lags=8)
ck(abs(_c["ic_ir_nw"] - _a["ic_ir_nw"]) > 1e-9, "lags 变化 ⇒ NW 版必须变化")


# ============================================================================
# \u2466 nw_lags \u7684 docstring \u91cc\u90a3\u51e0\u4e2a\u6570\u5b57\u5fc5\u987b\u662f\u771f\u7684
#   \u65e7 docstring \u5199\u7740\u300cT=552 \u21d2 6 \u671f\u3001T=1000 \u21d2 7 \u671f\u300d\uff0c\u5b9e\u9645\u662f 5 \u548c 6\u3002
#   \u8fd9\u79cd\u9519\u4e0d\u5f71\u54cd\u8fd0\u884c\uff0c\u4f46\u4f1a\u8ba9\u8bfb\u8005\u6309\u9519\u7684\u9884\u671f\u53bb\u770b\u62a5\u544a\u91cc\u7684 lag\u3002
#   \u6240\u4ee5\u6d4b\u8bd5\u5fc5\u987b\u3010\u76f4\u63a5\u8bfb docstring\u3011\u53bb\u6bd4\uff0c\u800c\u4e0d\u662f\u91cd\u7b97\u4e00\u904d\u516c\u5f0f\u3002
# ============================================================================
print()
print("\u2500\u2500 \u2466 nw_lags \u7684 docstring \u6570\u5b57\u2500\u2500")
import inspect, re as _re  # noqa: E402
from backtest.metrics import nw_lags as _nwl  # noqa: E402
_doc = inspect.getdoc(_nwl) or ""
_pairs = _re.findall(r"T=(\d+)\s*\u21d2\s*(\d+)\s*\u671f", _doc)
ck(len(_pairs) == 3, f"docstring \u91cc\u5e94\u8be5\u5217\u4e3e 3 \u7ec4\u6837\u672c\u91cf\uff08\u5b9e\u6d4b {len(_pairs)}\uff09")
for _t, _expect in _pairs:
    _got = _nwl(int(_t))
    ck(_got == int(_expect),
       f"docstring \u5199 T={_t} \u21d2 {_expect} \u671f\uff0c\u5b9e\u9645 nw_lags({_t}) = {_got}")

passed = sum(1 for c in CK if c)
print("\n" + "=" * 46)
print("  \u7ed3\u679c: \u901a\u8fc7 %d / %d" % (passed, len(CK)))
if passed != len(CK):
    print("  \u4ecd\u6709 %d \u9879\u5931\u8d25 \u274c" % (len(CK) - passed))
    # PITFALLS #8：永远绿的测试等于没有测试。返回码必须非 0，否则 CI / 脚本里失败永远不报警。
    sys.exit(1)
