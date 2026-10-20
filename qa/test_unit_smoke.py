"""不依赖任何数据的单元测试（CI 上跑这一批）。

为什么单独一个文件：qa/test_acceptance.py 里的 17 个验收全都要真面板（727 MB），
CI 上跑不了。这里只测【纯函数】，用随机数当场生成输入，和 pandas 对拍。

覆盖的是 2026-10-19 重写的两个核心函数：
  · features._roll   —— 替代 pandas .rolling()（mean / std）
  · metrics.rank1d   —— 替代 Series.rank()（ties=first / average）

这两个是"大数组写入权"改造的重点，也是最容易改错的地方，所以必须有对拍。
"""
import numpy as np
import pandas as pd
import pytest

from backtest import metrics as MT
from backtest.features import _roll


@pytest.mark.parametrize("how", ["mean", "std"])
@pytest.mark.parametrize("n", [2, 5, 20])
def test_roll_matches_pandas(how, n):
    """_roll 必须和 pandas.rolling 逐格一致（含 NaN / inf 的处理）。"""
    rng = np.random.default_rng(12345)
    x = rng.normal(size=(120, 7)).astype("float64")
    # 制造不规则：整段 NaN、单个 NaN、0（amihud 的 amount==0 会出 inf）
    x[0:3, 0] = np.nan
    x[10, 1] = np.nan
    x[50:60, 2] = np.nan
    x[30, 3] = 0.0

    got = _roll(x, n, how)

    s = pd.DataFrame(x)
    want = (s.rolling(n, min_periods=n).mean() if how == "mean"
            else s.rolling(n, min_periods=n).std()).to_numpy()

    # NaN 结构必须完全一致（这是最关键的：预热区 + 窗口内有 NaN 的位置）
    assert np.array_equal(np.isnan(got), np.isnan(want)), "NaN 结构不一致"
    m = ~np.isnan(want)
    # 容差说明：cumsum 版方差是 E[x²]−E[x]²，有固有的抵消误差。
    # 实测（120×7 随机数）：n=2 时 max|rel|=1.7e-8（且只出现在 std≈4e-5 的极小值上），
    # n>=5 时 max|rel|<5e-14。本项目只用 n=5/20/120，所以 rtol=1e-7 足够严。
    assert np.allclose(got[m], want[m], rtol=1e-7, atol=1e-10), (
        f"数值不一致: max|abs|={np.max(np.abs(got[m] - want[m])):.3e}")


@pytest.mark.parametrize("n", [1, 3, 20])
def test_roll_inf_is_invalid(n):
    """inf 必须当作无效值（否则 cumsum 里 inf-inf 会污染后面所有窗口）。"""
    x = np.ones((40, 3))
    x[5, 0] = np.inf
    out = _roll(x, n, "mean")
    # 含 inf 的窗口必须全是 NaN（第 5 行往后的 n-1 个窗口）
    assert np.isnan(out[5:5 + n, 0]).all()


@pytest.mark.parametrize("ties", ["first", "average"])
@pytest.mark.parametrize("size", [5, 50, 4000])
def test_rank1d_matches_pandas(ties, size):
    """rank1d 必须和 pandas 的 rank 逐格一致（含大量并列）。"""
    rng = np.random.default_rng(7)
    x = rng.integers(0, max(2, size // 10), size=size).astype("float64")  # 大量并列
    got = MT.rank1d(x, ties=ties)
    want = pd.Series(x).rank(method=ties).to_numpy()
    assert np.allclose(got, want), f"max|diff| = {np.max(np.abs(got - want)):.3e}"


def test_rank1d_first_is_a_permutation():
    """ties='first' 用在外层分层：结果必须是 1..n 的一个排列（不重不漏）。"""
    rng = np.random.default_rng(11)
    x = np.zeros(500)  # 全并列 —— 最坏情况
    r = MT.rank1d(x, ties="first")
    assert np.array_equal(np.sort(r), np.arange(1, 501, dtype="float64"))
