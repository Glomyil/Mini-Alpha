"""两类数据源的行为测试（2026-11-18）。

结构类断言（登记表形状 / 目录唯一来源 / 契约与产出对齐）在 qa/test_static_guards.py；
这里只放【行为】：kind 从 loader 一路传到环节，各层表现对不对。

三个 seam：
  base.check_required_columns      —— 契约分派，未知 kind 立刻报错
  align.run(..., kind=...)         —— 时间列按 kind 切，去重排序在正确的轴上
  pipeline.run_one(..., kind=...)  —— 把 kind 原样转给每个环节（断了不报错，所以必须测）
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from data import loader
from data.cleaner import align, base, pipeline


# ── loader / base：契约分派 ──────────────────────────────────

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


# ── align：时间列按 kind 切 ─────────────────────────────────

def _stock():
    """行情：乱序 + 一条重复 date + 一条坏日期。"""
    return pd.DataFrame({
        "date": ["2020-01-03", "2020-01-02", "2020-01-02", "坏日期", "2020-01-06"],
        "open": [3, 2, 2, 9, 6], "high": [3, 2, 2, 9, 6],
        "low": [3, 2, 2, 9, 6], "close": [3, 2, 2, 9, 6],
        "volume": [30, 20, 20, 90, 60],
    })


def _fund():
    """财务：乱序 + 一条重复 statDate + 一条坏日期；列叫 statDate 不叫 date。

    重复的两条 roeAvg 故意不同（2.0 / 2.5），否则测不出 keep="first" 到底保哪条。
    """
    return pd.DataFrame({
        "statDate": ["2020-06-30", "2020-03-31", "2020-03-31", "坏日期", "2020-09-30"],
        "pubDate_max": ["2020-08-28", "2020-04-30", "2020-04-30", "2020-10-31", "2020-10-31"],
        "roeAvg": [3.0, 2.0, 2.5, 9.0, 4.0],
    })


def test_default_kind_is_stock_and_unchanged():
    """不传 kind ⇒ 走 date 列，原有调用者的行为一点不变。"""
    out = align.run(_stock(), {}, {})
    assert out.attrs["align"]["kind"] == "stock"
    assert out.attrs["align"]["time_col"] == "date"
    assert len(out) == 3, "5 行里：1 坏日期 + 1 重复 ⇒ 剩 3"
    assert out["date"].is_monotonic_increasing


def test_fundamental_kind_switches_to_statdate():
    """kind="fundamental" ⇒ 认 statDate，不要求 date 列存在。"""
    out = align.run(_fund(), {}, {}, kind="fundamental")
    assert out.attrs["align"]["kind"] == "fundamental"
    assert out.attrs["align"]["time_col"] == "statDate"
    assert len(out) == 3
    assert out["statDate"].is_monotonic_increasing


def test_dedup_keeps_first_on_the_right_axis():
    """去重是 keep="first"，且必须发生在 statDate 轴上（不是 date）。"""
    out = align.run(_fund(), {}, {}, kind="fundamental")
    row = out[out["statDate"] == pd.Timestamp("2020-03-31")].iloc[0]
    assert row["roeAvg"] == 2.0, "保留原始顺序里的第一条（2.0），不是后面的 2.5"


def test_bad_date_rows_are_counted_in_attrs():
    """坏日期行被删，且删了几条要记账（不是静默丢）。"""
    rec = align.run(_fund(), {}, {}, kind="fundamental").attrs["align"]
    assert rec["rows_in"] == 5
    assert rec["bad_date_rows_dropped"] == 1
    assert rec["dup_rows_dropped"] == 1
    assert rec["rows_out"] == 3


def test_align_unknown_kind_raises_before_touching_data():
    """未知 kind 当场报错，列出可选项，不许回落。"""
    with pytest.raises(ValueError, match="未知的数据类型"):
        align.run(_stock(), {}, {}, kind="industry")


def test_wrong_kind_is_caught_by_the_contract():
    """财务表当行情喂 ⇒ 缺 date 列，当场 ValueError（不是静默出空表）。"""
    with pytest.raises(ValueError, match="缺了这些必须列"):
        align.run(_fund(), {}, {})            # 忘了传 kind


# ── pipeline.run_one：kind 必须原样往下转 ────────────────────

def _recorder(seen):
    """假环节：只记录收到了什么 kind，原样返回 df。"""
    def _run(df, params, ctx, kind="stock"):
        seen.append(kind)
        return df
    return SimpleNamespace(run=_run)


def test_run_one_forwards_kind_to_every_stage():
    """断了不会有任何报错 —— 行情照跑、测试照过，只有财务链会在某处莫名缺列。"""
    seen = []
    steps = [("a", _recorder(seen)), ("b", _recorder(seen))]
    ctx = {"asset_type": "stock", "code": "sh.600000"}
    pipeline.run_one(_stock(), ctx, steps=steps, kind="fundamental")
    assert seen == ["fundamental", "fundamental"]


def test_run_one_defaults_to_stock():
    """不传 kind ⇒ 环节收到 "stock"，现有调用零影响。"""
    seen = []
    steps = [("a", _recorder(seen))]
    pipeline.run_one(_stock(), {"asset_type": "stock", "code": "x"}, steps=steps)
    assert seen == ["stock"]