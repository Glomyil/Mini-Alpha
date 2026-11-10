"""tests/test_neutralize.py —— 中性化（T-6.2）的回归网

跑法（项目根目录）：
    python tests/test_neutralize.py

两个 seam：
    S1  neutralize(...)            算法唯一公开接口
    S2  load_spec() 的公共属性      配置层的唯一入口

不测：main.py 的接线顺序（要跑全流水线，交给压力测试）

⚠️ 关于「三条恒等式」的定位：
    它们是 OLS 法方程 Xᵀr=0 的直接推论 ⇒ 【不是】算法正确性的证明。
    真正的作用是【探测器】：mask 用错、残差写回了错误的列、忘了 drop_first，
    这三条会立刻破 ⇒ 用它们抓接线错误，不用它们证明算得对。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.neutralize import neutralize  # noqa: E402
from config.config_manager import config  # noqa: E402
from config.research_spec import load_spec  # noqa: E402

CK = []


def ck(cond, msg):
    CK.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


def _panel(factors, industry, mv):
    """把三个 list-of-list 组装成 factor + panels（同索引同列）"""
    cols = [f"s{i}" for i in range(len(factors[0]))]
    idx = pd.DatetimeIndex([f"2020-01-{d:02d}" for d in range(2, 2 + len(factors))])
    return (pd.DataFrame(factors, index=idx, columns=cols),
            {"industry": pd.DataFrame(industry, index=idx, columns=cols).astype("int16"),
             "float_mv": pd.DataFrame(mv, index=idx, columns=cols)})


# ============================================================================
# S1-① 手算小面板（期望值来自手算，不是用实现重算）
#   4 只票 / 2 个行业 / 市值全相同 ⇒ ln(mv) 是常数，被截距吸收
#   因子 [1,3,2,6] ⇒ 横截面百分位秩 [0.25,0.75,0.5,1.0]
#   行业 A={0.25,0.75} 均值 0.5  ⇒ 残差 -0.25, +0.25
#   行业 B={0.5, 1.0 } 均值 0.75 ⇒ 残差 -0.25, +0.25
# ============================================================================
print("\n\u2500\u2500 S1-\u2460 \u624b\u7b97\u5c0f\u9762\u677f\u2500\u2500")
f1, p1 = _panel([[1.0, 3.0, 2.0, 6.0]], [[1, 1, 2, 2]], [[1e10] * 4])
r1 = neutralize(f1, p1, [f1.index[0]], min_n=2).loc[f1.index[0]].to_list()
ck(np.allclose(r1, [-0.25, 0.25, -0.25, 0.25], atol=1e-12),
   f"\u6b8b\u5dee {np.round(r1, 6).tolist()} == \u624b\u7b97 [-0.25,0.25,-0.25,0.25]")

# 因子恰好等于行业均值 ⇒ 被完全解释 ⇒ 残差全 0
f2, _ = _panel([[2.0, 2.0, 4.0, 4.0]], [[1, 1, 2, 2]], [[1e10] * 4])
r2 = neutralize(f2, p1, [f2.index[0]], min_n=2).to_numpy()
ck(np.abs(r2).max() < 1e-12, f"\u56e0\u5b50=\u884c\u4e1a\u5747\u503c \u21d2 \u6b8b\u5dee\u5168 0\uff08\u5b9e\u6d4b {np.abs(r2).max():.1e}\uff09")


# ============================================================================
# S1-② 三条法方程恒等式（探测器：抓 mask / 写回 / drop_first 的错误）
# ============================================================================
print("\n\u2500\u2500 S1-\u2461 \u4e09\u6761\u6052\u7b49\u5f0f\u2500\u2500")
rng = np.random.default_rng(42)
N, T = 300, 2
mv = np.exp(rng.normal(22, 1.5, (T, N)))
ind = rng.integers(1, 5, (T, N))
fac = 0.8 * np.log(mv) + rng.normal(0, 1, (T, N))   # 故意与市值强相关
f3, p3 = _panel(fac.tolist(), ind.tolist(), mv.tolist())
r3 = neutralize(f3, p3, list(f3.index), min_n=10)

_worst_corr = _worst_mean = _worst_r2 = 0.0
for d in f3.index:
    i = f3.index.get_loc(d)
    rr = r3.loc[d].to_numpy(dtype="float64")
    lnmv = np.log(mv[i]); ids = ind[i]
    ok = np.isfinite(rr)
    a, b, k = rr[ok], lnmv[ok], ids[ok]
    _worst_corr = max(_worst_corr, abs(float(np.corrcoef(a, b)[0, 1])))
    for g in np.unique(k):
        _worst_mean = max(_worst_mean, abs(float(a[k == g].mean())))
    X = np.column_stack([np.ones(len(a))]
                        + [(k == g).astype(float) for g in np.unique(k)[1:]]
                        + [b])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    e = a - X @ beta
    sst = float(((a - a.mean()) ** 2).sum())
    if sst > 0:
        _worst_r2 = max(_worst_r2, 1.0 - float((e ** 2).sum()) / sst)
ck(_worst_corr < 1e-10, f"\u2460 |corr(\u6b8b\u5dee, ln\u5e02\u503c)| = {_worst_corr:.1e}")
ck(_worst_mean < 1e-10, f"\u2461 \u884c\u4e1a\u5185\u90e8 |mean(\u6b8b\u5dee)| = {_worst_mean:.1e}")
ck(_worst_r2 < 1e-10, f"\u2462 \u518d\u56de\u5f52 R\u00b2 = {_worst_r2:.1e}")


# ============================================================================
# S1-③ 形状契约 + NaN 契约（需求规定，不是实现细节）
# ============================================================================
print("\n\u2500\u2500 S1-\u2462 \u5f62\u72b6 / NaN \u5951\u7ea6\u2500\u2500")
f4, p4 = _panel([[1.0, np.nan, 3.0] + [4.0] * 9,
                 [2.0] * 12,
                 [3.0] * 12],
                [[1, 1, 0] + [0] * 9, [2] * 12, [2] * 12],
                [[1e10] * 12, [1e10] * 12, [1e10] * 12])
mv4 = [[1e10] * 12, [1e10] * 12, [1e10] * 12]
mv4[0][2] = np.nan                      # s2 没市值
p4["float_mv"] = pd.DataFrame(mv4, index=f4.index, columns=f4.columns)
# 只传第 0 期和第 2 期作 rdates ⇒ 第 1 期就该全是 NaN
r4 = neutralize(f4, p4, [f4.index[0], f4.index[2]], min_n=2)
ck(r4.shape == f4.shape and list(r4.columns) == list(f4.columns),
   f"\u5f62\u72b6/\u5217\u4e0e\u8f93\u5165\u4e00\u81f4\uff08\u5b9e\u6d4b {r4.shape}\uff09")
ck(r4.index.equals(f4.index), "\u7d22\u5f15\u4e0e\u8f93\u5165\u4e00\u81f4")
ck(np.isnan(r4.loc[f4.index[0], "s1"]), "\u56e0\u5b50\u662f NaN \u21d2 \u6b8b\u5dee\u662f NaN\uff08\u4e0d\u662f 0\uff09")
ck(np.isnan(r4.loc[f4.index[0], "s2"]), "\u65e0\u5e02\u503c \u21d2 \u6b8b\u5dee NaN")
ck(not np.isnan(r4.loc[f4.index[0], "s3"]), "industry=0 \u7684\u7968\u4ecd\u53c2\u4e0e\uff08\u4e0d\u4e22\u6837\u672c\uff09")

# 非 rdates 行必须全 NaN（0 表示"因子值为 0"，绝不能填 0）
ck(np.isnan(r4.loc[f4.index[1]].to_numpy()).all(), "\u975e rdates \u884c\u5168 NaN")

# 有效样本 < min_n ⇒ 该期不算（不崩、留 NaN）
r5 = neutralize(f4, p4, [f4.index[0]], min_n=999)
ck(np.isnan(r5.to_numpy()).all(), "\u6709\u6548\u6837\u672c < min_n \u21d2 \u8be5\u671f\u5168 NaN\uff08\u4e0d\u5d29\uff09")


# ============================================================================
# S1-④ 参数校验（宁可报错，不许静默用默认）
# ============================================================================
print("\n\u2500\u2500 S1-\u2463 \u53c2\u6570\u6821\u9a8c\u2500\u2500")


def _raises(fn, label):
    try:
        fn()
        ck(False, label + " \u21d2 \u7adf\u7136\u6ca1\u62a5\u9519")
    except Exception as e:
        ck(True, f"{label} \u21d2 {type(e).__name__}")


_raises(lambda: neutralize(f1, p1, [f1.index[0]], exposures=("industry", "ln_float_mvv"), min_n=2),
        "\u672a\u77e5\u66b4\u9732\u9879 ln_float_mvv")
_raises(lambda: neutralize(f1, {"industry": p1["industry"]}, [f1.index[0]], min_n=2),
        "panels \u7f3a float_mv")
_raises(lambda: neutralize(f1, {"float_mv": p1["float_mv"]}, [f1.index[0]], min_n=2),
        "panels \u7f3a industry")


# ============================================================================
# S2 配置层：yml 能穿透 + 指纹 12/17 段 + banner 不丢信息
# ============================================================================
print("\n\u2500\u2500 S2 \u914d\u7f6e\u5c42\u2500\u2500")

# ① yml 必须真的被读到（缩进挂错到顶层时，这里会是 None ⇒ 抓得住）
_raw = config.get("research.neutralize.enabled")
ck(_raw is not None, f"research.neutralize.enabled \u80fd\u4ece yml \u8bfb\u5230\uff08\u5b9e\u6d4b {_raw!r}\uff09")
ck(isinstance(_raw, bool), f"enabled \u662f\u771f\u5e03\u5c14\uff08\u5b9e\u6d4b {type(_raw).__name__}\uff09")

sp = load_spec()
ck(sp.neutralize == _raw, f"RunSpec.neutralize \u4e0e yml \u4e00\u81f4\uff08{sp.neutralize}\uff09")

# ② 指纹：关闭 12 段 / 开启 17 段   （独立依据：设计规定 12 + 5）
_fp = sp.fingerprint()
ck(isinstance(_fp, str), "fingerprint() \u8fd4\u56de\u5b57\u7b26\u4e32\uff08\u4e0d\u662f None\uff09")
# \u26a0\ufe0f \u4e0d\u80fd\u5199\u6b7b\u300c\u5173\u95ed\u65f6 12 \u6bb5\u300d\uff1ayml \u91cc\u7684 enabled \u968f\u65f6\u4f1a\u5207\uff08\u8dd1\u5bf9\u7167\u5b9e\u9a8c\u5c31\u8981\u5207\uff09\uff0c
#    \u6d4b\u8bd5\u8ddf\u7740 yml \u53d8\u7ea2\u5c31\u53d8\u6210\u300c\u6d4b\u914d\u7f6e\u300d\u800c\u4e0d\u662f\u300c\u6d4b\u884c\u4e3a\u300d\u3002
#    \u21d2 \u6309 sp.neutralize \u7684\u771f\u5b9e\u503c\u63a8\u51fa\u5e94\u6709\u7684\u6bb5\u6570\uff0c\u884c\u4e3a\u5224\u636e\u4e0d\u53d8\u3002
_seg_should = 17 if sp.neutralize else 12
ck(len(_fp.split("|")) == _seg_should,
   f"\u6307\u7eb9\u6bb5\u6570\u8ddf enabled \u8d70\uff1aenabled={sp.neutralize} \u21d2 \u5e94 {_seg_should} \u6bb5\uff08\u5b9e\u6d4b {len(_fp.split('|'))}\uff09")
_on = load_spec(neutralize=True)
ck(len(_on.fingerprint().split("|")) == 17,
   f"\u5f00\u542f\u65f6 17 \u6bb5\uff08\u5b9e\u6d4b {len(_on.fingerprint().split('|'))}\uff09")
ck(_on.cfg_hash != load_spec(neutralize=False).cfg_hash,
   "\u5f00\u542f vs \u5173\u95ed \u21d2 cfg_hash \u4e0d\u540c")

# 只剥行业 ⇒ hash 再变（证明 exposures 真的进了指纹）
_one = load_spec(neutralize=True, neutralize_exposures=("industry",))
ck(_one.cfg_hash != _on.cfg_hash, "exposures \u53d8\u5316 \u21d2 hash \u53d8\u5316")

# ③ banner：关闭时【不能】只剩一行（P1-7 那个优先级坑的回归）
_b_sp = sp.banner()
ck(sp.cfg_hash in _b_sp, f"banner \u59cb\u7ec8\u542b spec hash\uff08\u5b9e\u6d4b {_b_sp[:28]!r}...\uff09")
# banner \u5fc5\u987b\u8ddf enabled \u8d70\uff1a\u5f00\u5c31\u5370\u5f00\u3001\u5173\u5c31\u5370\u5173\uff08\u4e0d\u5199\u6b7b\u4efb\u4f55\u4e00\u8fb9\uff09
ck(("\u4e2d\u6027\u5316:\u5f00" in _b_sp) if sp.neutralize else ("\u4e2d\u6027\u5316:\u5173" in _b_sp),
   f"banner \u7684\u4e2d\u6027\u5316\u72b6\u6001\u8ddf enabled \u4e00\u81f4\uff08enabled={sp.neutralize}\uff09")
_b_on = _on.banner()
ck("\u4e2d\u6027\u5316:\u5f00" in _b_on and "\u79e9" in _b_on,
   "\u5f00\u542f\u65f6 banner \u5370\u51fa\u66b4\u9732\u9879 + \u5c3a\u5ea6\uff08\u5426\u5219\u4e24\u5957\u914d\u7f6e\u7684\u62a5\u544a\u957f\u5f97\u4e00\u6837\uff09")


# ============================================================================
# S1-⑤ exposures 必须真的决定【剥哪几项】（2026-11-10 实测发现的 bug：
#       exposures 只被用来校验，设计矩阵却无条件把行业和市值都加进去
#       ⇒ 三种写法跑出完全相同的残差，而 banner 却印着不同的暴露项 ⇒ 口径撒谎）
#
# 判据用法方程的两条恒等式 + 「没被剥的那一项必须还活着」：
#   只剥行业     ⇒ 行业内均值 ≈ 0（剥了）  且 |corr(残差, ln市值)| 明显 > 0（没剥市值）
#   只剥市值     ⇒ |corr| ≈ 0（剥了）      且 行业内 |均值| 明显 > 0（没剥行业）
#   两个都剥     ⇒ 两者都 ≈ 0
#   空列表       ⇒ ValueError（开着中性化却一项都不剥 ⇒ 等于只去均值，不许静默）
# ============================================================================
print("\n\u2500\u2500 S1-\u2464 exposures \u771f\u7684\u751f\u6548\u5417\u2500\u2500")

_NG = 200                                  # 每个行业的票数
_rng5 = np.random.default_rng(7)
_lnmv5 = _rng5.normal(23.0, 1.2, 2 * _NG)  # 两个行业共用同一组市值 ⇒ 行业与市值相互独立
_ind5 = np.array([1] * _NG + [2] * _NG)
# 因子 = 强规模效应 + 强行业效应 + 小噪声（两个效应都要能被抓住）
_fac5 = 1.0 * _lnmv5 + 2.0 * (_ind5 == 2) + _rng5.normal(0, 0.2, 2 * _NG)
_f5, _p5 = _panel([_fac5.tolist()], [_ind5.tolist()], [np.exp(_lnmv5).tolist()])


def _ortho(exposures):
    """返回 (max|corr(残差, ln市值)|, max 行业内|均值|)"""
    r = neutralize(_f5, _p5, [_f5.index[0]], exposures=exposures, min_n=10)
    a = r.loc[_f5.index[0]].to_numpy(dtype="float64")
    b = _lnmv5
    ok = np.isfinite(a)
    a, b, k = a[ok], b[ok], _ind5[ok]
    c = abs(float(np.corrcoef(a, b)[0, 1]))
    m = max(abs(float(a[k == g].mean())) for g in np.unique(k))
    return c, m


_c_both, _m_both = _ortho(("industry", "ln_float_mv"))
ck(_c_both < 1e-8 and _m_both < 1e-8,
   f"\u4e24\u4e2a\u90fd\u5265 \u21d2 \u4e24\u4e2a\u6052\u7b49\u5f0f\u90fd\u6210\u7acb\uff08corr={_c_both:.1e}, mean={_m_both:.1e}\uff09")

_c_ind, _m_ind = _ortho(("industry",))
ck(_m_ind < 1e-8, f"\u53ea\u5265\u884c\u4e1a \u21d2 \u884c\u4e1a\u5185\u5747\u503c\u2248 0\uff08{_m_ind:.1e}\uff09")
ck(_c_ind > 0.3, f"\u53ea\u5265\u884c\u4e1a \u21d2 \u5e02\u503c\u3010\u6ca1\u88ab\u5265\u3011\uff0c|corr| \u5e94\u660e\u663e > 0\uff08\u5b9e\u6d4b {_c_ind:.3f}\uff09")

_c_mv, _m_mv = _ortho(("ln_float_mv",))
ck(_c_mv < 1e-8, f"\u53ea\u5265\u5e02\u503c \u21d2 |corr|\u2248 0\uff08{_c_mv:.1e}\uff09")
ck(_m_mv > 1e-3, f"\u53ea\u5265\u5e02\u503c \u21d2 \u884c\u4e1a\u3010\u6ca1\u88ab\u5265\u3011\uff0c\u884c\u4e1a\u5185\u5747\u503c\u5e94\u660e\u663e \u2260 0\uff08\u5b9e\u6d4b {_m_mv:.4f}\uff09")

_raises(lambda: neutralize(_f5, _p5, [_f5.index[0]], exposures=(), min_n=10),
        "空 exposures")


passed = sum(1 for c in CK if c)
print("\n" + "=" * 46)
print("  \u7ed3\u679c: \u901a\u8fc7 %d / %d" % (passed, len(CK)))
if passed != len(CK):
    print("  \u4ecd\u6709 %d \u9879\u5931\u8d25 \u274c" % (len(CK) - passed))
