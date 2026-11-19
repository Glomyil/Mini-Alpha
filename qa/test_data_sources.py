"""数据源读取的行为测试（2026-11-18）。

结构类断言（登记表形状 / 目录唯一来源 / 契约与产出对齐）在 qa/test_static_guards.py；
这里只放【行为】：调用出错时怎么表现。
"""
from __future__ import annotations

import pandas as pd
import pytest

from data import loader
from data.cleaner import base


def test_unknown_kind_raises_instead_of_falling_back():
    """拼错 kind 必须立刻报错，不许静默回落到 stock（那会变成用错契约）。"""
    with pytest.raises(ValueError, match="未知的数据类型"):
        loader.list_codes(kind="fundmental")        # 少个 a
    with pytest.raises(ValueError, match="未知的数据类型"):
        loader.load_one("sh.600000", "2020-01-01", "2020-12-31", kind="fundmental")


def test_forgetting_kind_misdiagnoses_as_missing_columns():
    """已知坑：直接拿财务表走默认契约，报出的是「缺 date/open/close」。

    故意钉住现状 —— required 的默认语义必须保持（6 处位置调用依赖它）。
    写在这里是为了让下次被这个报错误导的人能搜到。
    """
    fund = pd.DataFrame({"statDate": [pd.Timestamp("2020-03-31")],
                         "pubDate_max": [pd.Timestamp("2020-04-30")]})
    with pytest.raises(ValueError) as ei:
        base.check_required_columns(fund)          # 没传 required
    msg = str(ei.value)
    assert "statDate" not in msg, "报的应该是行情契约的缺列，不是财务的"
    assert "date" in msg and "close" in msg, f"实际信息：{msg}"