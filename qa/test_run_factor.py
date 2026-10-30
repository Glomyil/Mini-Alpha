"""backtest/pipeline.run_factor 的契约测试。

⚠️ 文件名不叫 test_pipeline.py —— tests/ 下已经有一个同名的，
   那份测的是 data/cleaner/pipeline.py（清洗层执行器），和这里无关。
   用被测函数名命名，歧义最少（test_scheduler.py 里已经为这个歧义写过注释）。

只测两条错误分支 —— 数字对不对不在这里测：
  · 数字  已在抽取时实测过（层0 -6.3878% / 层4 5.1765%，和台账一致）
  ·        而且每次跑 main.py 都会记账，台账会持续盯着
  ·        写进 CI 要 727 MB 面板，不可能

所以这里测的是「库该在什么时候吵」。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.factors import FACTORS                      # noqa: E402
from backtest.pipeline import run_factor                  # noqa: E402


def test_unknown_factor_raises_keyerror():
    """因子名不存在 => KeyError，且提示里带上可用的名字。

    检查发生在 build_features 之前，所以传空 dict / 空 list 就够了 —— 不碰真数据。
    """
    with pytest.raises(KeyError) as ei:
        run_factor("no_such_factor", {}, [])
    msg = str(ei.value)
    assert "no_such_factor" in msg, "报错该点名写错的那个因子"
    assert "rev_20d" in msg, "报错该列出可用因子，不然还得自己去翻 FACTORS"


def test_missing_panel_field_raises_keyerror():
    """面板缺 requires 声明的字段 => 在做任何计算之前就 KeyError。

    这里给的 panels 只有一个 hfq_close，缺 amount —— 照样能测，
    因为 requires 检查在读数据之前。
    """
    fact = "__tmp_missing_field__"
    FACTORS[fact] = {"fn": lambda f, p: f["ret_5"], "sign": 1,
                     "needs": [], "requires": ["amount"]}
    try:
        with pytest.raises(KeyError) as ei:
            run_factor(fact, {"hfq_close": None}, [])
        assert "amount" in str(ei.value), "报错该点名缺哪个字段"
    finally:
        del FACTORS[fact]        # ★ 必须清理，否则污染其他测试
