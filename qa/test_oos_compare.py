"""T-3.8 的 OOS 工具：形状重合 ρ 与它的 bootstrap 单侧 p。

只测这一个 seam（_rho_and_p）：它是五条判据里唯一一处"统计量本身要自己算"的地方，
上游 IC / 多空 t 都直接调 metrics 里的现成实现。

期望值来源：全部由【人工构造的矩阵】给出，不从被测代码反推。
"""
from __future__ import annotations

import numpy as np
import pytest

from ops.oos_compare import _rho_and_p


def test_identical_shape_gives_rho_one_and_zero_p():
    """两段完全一样 ⇒ ρ 恰好 1，且单侧 p 恰好 0（所有重采样都 >0）。"""
    R = np.array([[0.01, 0.02, 0.03],
                  [0.02, 0.04, 0.06],
                  [0.00, 0.01, 0.02]])
    rho, p, lo, hi = _rho_and_p(R, R)
    assert rho == pytest.approx(1.0, abs=1e-12)
    assert p == 0.0
    assert lo <= hi


def test_reversed_shape_gives_minus_one_and_p_one():
    """第二段把层序倒过来 ⇒ ρ 恰好 −1，单侧 p 恰好 1。

    这条同时钉住了"p 是单侧"：双侧检验下 |ρ|=1 会给出 p≈0，
    而我们要问的是"ρ 显著为正吗"，所以反序必须是 p=1 而不是 p=0。
    """
    R1 = np.array([[0.01, 0.02, 0.03],
                   [0.02, 0.05, 0.08]])
    R2 = R1[:, ::-1]
    rho, p, lo, hi = _rho_and_p(R1, R2)
    assert rho == pytest.approx(-1.0, abs=1e-12)
    assert p == 1.0


def test_positive_linear_rescaling_does_not_change_rho():
    """ρ 对正数线性变换不变：R2 = a·R1 (a>0) ⇒ ρ 仍为 1。"""
    R1 = np.array([[0.01, 0.02, 0.03],
                   [0.03, 0.01, 0.02]])
    R2 = 10.0 * R1
    rho, _, _, _ = _rho_and_p(R1, R2)
    assert rho == pytest.approx(1.0, abs=1e-12)


def test_negative_scaling_flips_rho():
    """乘负号 ⇒ ρ 翻号。用来证明 ρ 不是只看量级的量。"""
    R1 = np.array([[0.01, 0.02, 0.03],
                   [0.03, 0.01, 0.02]])
    rho, _, _, _ = _rho_and_p(R1, -3.0 * R1)
    assert rho == pytest.approx(-1.0, abs=1e-12)


def test_degenerate_shape_raises_instead_of_returning_nan():
    """层收益全相等的段：ρ 数学上未定义。

    ⚠️ 必须抛错，不能返回 NaN —— 因为下游判据是 p_rho < 0.05，
    NaN 比较恒为 False，会【静默】把因子判成"未复现"，而真正的问题是数据退化。
    """
    R = np.zeros((5, 4))          # 每段每层收益都是 0 ⇒ 列方差为 0
    with pytest.raises(ValueError):
        _rho_and_p(R, R)


def test_rho_and_ci_are_in_range():
    """随机但有一定结构的输入：ρ、CI 必须都落在 [−1, 1]。"""
    rng = np.random.default_rng(0)
    base = rng.normal(size=6)
    R1 = base + 0.3 * rng.normal(size=(40, 6))
    R2 = base + 0.3 * rng.normal(size=(70, 6))
    rho, p, lo, hi = _rho_and_p(R1, R2)
    for v in (rho, lo, hi):
        assert -1.0 - 1e-12 <= v <= 1.0 + 1e-12
    assert 0.0 <= p <= 1.0
    assert lo <= hi
