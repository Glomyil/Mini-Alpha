"""pipeline 双链分支的测试。

seam：`run_one(df, ctx, steps=None, kind)` —— 唯一对外接口。

两层：
  第一层 注入假环节，只验【分派】：kind 选中哪条清单、steps 覆盖是否生效
  第二层 真实环节 + 注入样例，验【流程真的跑过】：财务样例出 43 列、行情样例走满 6 格
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cleaner import pipeline as PL
from data.fundamental_merge import TABLES

_CTX = {"universe": "stock", "asset_type": "stock", "code": "sh.600000"}


class _Spy:
    """假环节：被调用时记下自己的名字，原样返回 df。"""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def run(self, df, params, ctx, kind):
        self.calls.append((self.name, kind))
        return df


# ── 第一层：分派 ───────────────────────────────────────
def test_kind_stock_runs_only_the_stock_chain(monkeypatch):
    """kind="stock" 只走 PIPELINE，一格都不碰 FUND_PIPELINE。"""
    calls = []
    monkeypatch.setattr(PL, "PIPELINE", [("s1", _Spy("s1", calls)), ("s2", _Spy("s2", calls))])
    monkeypatch.setattr(PL, "FUND_PIPELINE", [("f1", _Spy("f1", calls))])

    PL.run_one(pd.DataFrame({"a": [1]}), dict(_CTX), kind="stock")
    assert [n for n, _ in calls] == ["s1", "s2"]


def test_kind_fundamental_runs_only_the_fund_chain(monkeypatch):
    """kind="fundamental" 只走 FUND_PIPELINE。"""
    calls = []
    monkeypatch.setattr(PL, "PIPELINE", [("s1", _Spy("s1", calls))])
    monkeypatch.setattr(PL, "FUND_PIPELINE", [("f1", _Spy("f1", calls)), ("f2", _Spy("f2", calls))])

    PL.run_one(pd.DataFrame({"a": [1]}), dict(_CTX), kind="fundamental")
    assert [n for n, _ in calls] == ["f1", "f2"]


def test_kind_is_passed_down_to_every_stage(monkeypatch):
    """kind 要原样传给每个环节 —— 环节内部靠它分派契约（如 align 的时间列）。"""
    calls = []
    monkeypatch.setattr(PL, "FUND_PIPELINE", [("f1", _Spy("f1", calls))])

    PL.run_one(pd.DataFrame({"a": [1]}), dict(_CTX), kind="fundamental")
    assert calls == [("f1", "fundamental")]


def test_steps_override_wins_over_kind(monkeypatch):
    """显式传 steps 时，kind 只影响传给环节的参数，不决定走哪条清单。"""
    calls = []
    monkeypatch.setattr(PL, "PIPELINE", [("s1", _Spy("s1", calls))])
    monkeypatch.setattr(PL, "FUND_PIPELINE", [("f1", _Spy("f1", calls))])
    only = [("x1", _Spy("x1", calls))]

    PL.run_one(pd.DataFrame({"a": [1]}), dict(_CTX), steps=only, kind="fundamental")
    assert [n for n, _ in calls] == ["x1"]
    assert calls[0][1] == "fundamental"


def test_unknown_kind_raises():
    """kind 拼错必须当场炸，不能静默走另一条链。"""
    with pytest.raises(ValueError, match="kind"):
        PL.run_one(pd.DataFrame({"a": [1]}), dict(_CTX), kind="fund")


def test_missing_ctx_key_raises():
    """ctx 缺 asset_type / code ⇒ 当场 KeyError。"""
    with pytest.raises(KeyError, match="code"):
        PL.run_one(pd.DataFrame({"a": [1]}), {"asset_type": "stock"}, kind="fundamental")


# ── 第二层：真实环节 + 注入样例 ────────────────────────
_IND = [c for cols in TABLES.values() for c in cols]


def _event(rows):
    """39 列事件宽表的最小合法样例。"""
    base = {c: np.nan for c in _IND}
    for t in TABLES:
        base[f"pubDate_{t}"] = pd.NaT
    recs = []
    for r in rows:
        d = dict(base)
        d.update(r)
        recs.append(d)
    cols = ["statDate"] + [f"pubDate_{t}" for t in TABLES] + _IND
    df = pd.DataFrame(recs, columns=cols)
    df["statDate"] = pd.to_datetime(df["statDate"])
    for t in TABLES:
        df[f"pubDate_{t}"] = pd.to_datetime(df[f"pubDate_{t}"])
    df["pubDate_max"] = df[[f"pubDate_{t}" for t in TABLES]].max(axis=1)
    return df.sort_values("statDate").reset_index(drop=True)


def test_fund_chain_turns_an_event_table_into_daily():
    """注入 2 期事件表 ⇒ 走完三个财务环节 ⇒ 2871 行 × 43 列。"""
    fund = _event([
        {"statDate": "2019-12-31", "pubDate_profit": "2020-04-25",
         "pubDate_max": "2020-04-25", "roeAvg": 0.2, "liabilityToAsset": 0.5,
         "assetToEquity": 2.0, "netProfit": 1.0},
        {"statDate": "2020-03-31", "pubDate_profit": "2020-04-25",
         "pubDate_max": "2020-04-25", "roeAvg": 0.05, "liabilityToAsset": 0.4,
         "assetToEquity": 1.6667, "netProfit": 0.3},
    ])
    days = pd.DatetimeIndex(pd.bdate_range("2020-04-24", periods=3))
    ctx = {**_CTX, "trading_days": days}

    out, reps = PL.run_one(fund, ctx, kind="fundamental")

    assert [r["name"] for r in reps] == ["align", "fund_quality", "fundamental_fill"]
    assert out.shape == (3, 43)
    assert reps[2]["metrics"]["rows_before"] == 2
    assert reps[2]["metrics"]["rows_after"] == 3
    assert np.isnan(out["roeAvg"].iloc[0]), "首份公告之前必须是 NaN"
    assert out["roeAvg"].iloc[1] == pytest.approx(0.05)


def test_stock_chain_runs_all_six_stages_on_an_injected_sample():
    """注入一段真实行情的切片 ⇒ 走满 6 格 ⇒ 加出 4 个估值列与派生列。"""
    from data import loader

    codes = loader.list_codes()
    if not codes:
        pytest.skip("没有 stock_daily 数据")
    raw = loader.load_one(codes[0], "2015-01-01", "2026-12-31")
    sample = raw.head(400).copy()

    out, reps = PL.run_one(sample, dict(_CTX), kind="stock")

    assert [r["name"] for r in reps] == [n for n, _ in PL.PIPELINE]
    assert len(reps) == 6
    assert len(out) == 400, "行情链不删行（这段样本没有坏日期）"
    for c in ("hfq_close", "hfq_open", "ret_1d_hfq", "float_mv",
              "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "back_adj_factor"):
        assert c in out.columns, c