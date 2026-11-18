"""fundamental_merge 的行为测试（T-6.2）。

只测一个 seam：build_events(out=..., src=...)，全部断言从【落盘文件】观察。
夹具是手写的假 parquet，期望值来自手算 / 真实数据实测，不用实现反推：
  · sh.600018 的 91 天差异（快报 01-10 / 年报 04-10）是真实数据里实测到的
  · 列名与列序是规格规定的（39 列、按表分块），故意不 import TABLES 以免变成同义反复
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.fundamental_merge import build_events

# ── 五张表的列（规格，硬编码；改了这里就说明规格变了，测试该红）──
COLS = {
    "profit": ["roeAvg", "npMargin", "gpMargin", "netProfit",
               "epsTTM", "MBRevenue", "totalShare", "liqaShare"],
    "growth": ["YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"],
    "balance": ["currentRatio", "quickRatio", "cashRatio",
                "YOYLiability", "liabilityToAsset", "assetToEquity"],
    "operation": ["NRTurnRatio", "NRTurnDays", "INVTurnRatio",
                  "INVTurnDays", "CATurnRatio", "AssetTurnRatio"],
    "cashflow": ["CAToAsset", "NCAToAsset", "tangibleAssetToAsset",
                 "ebitToInterest", "CFOToOR", "CFOToNP", "CFOToGr"],
}


def _row(table: str, code: str, pub: str, stat: str, **over) -> dict:
    """造一行：三个键 + 该表全部指标（默认 "1.0"），用 over 覆盖特定列。"""
    r = {"code": code, "pubDate": pub, "statDate": stat}
    for c in COLS[table]:
        r[c] = "1.0"
    r.update(over)
    return r


def _write(src, table: str, code: str, rows: list[dict]) -> None:
    d = src / table
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{code}.parquet", index=False)


@pytest.fixture
def src(tmp_path):
    """三只票：A 齐全但有一格空串；B 缺 growth 一期 + 五表 pubDate 差 91 天；C 有一张表 0 字节。"""
    s = tmp_path / "fundamental"

    # A：三期齐全
    for t in COLS:
        rows = [_row(t, "sh.600000", "2020-04-10", sd) for sd in ("2019-12-31", "2020-03-31", "2020-06-30")]
        _write(s, t, "sh.600000", rows)
    # A 的 balance 在 2019-12-31 这一格是空串（不是 NULL）
    _write(s, "balance", "sh.600000", [
        _row("balance", "sh.600000", "2020-04-10", "2019-12-31", currentRatio=""),
        _row("balance", "sh.600000", "2020-04-10", "2020-03-31"),
        _row("balance", "sh.600000", "2020-04-10", "2020-06-30"),
    ])

    # B：growth 只有两期（缺 2020-03-31）；2019-12-31 五表公告日差 91 天
    for t in COLS:
        sds = ("2019-12-31", "2020-06-30") if t == "growth" else ("2019-12-31", "2020-03-31", "2020-06-30")
        rows = []
        for sd in sds:
            pub = "2020-01-10" if (sd == "2019-12-31" and t == "growth") else "2020-04-10"
            rows.append(_row(t, "sh.600018", pub, sd))
        _write(s, t, "sh.600018", rows)

    # C：balance 是 0 字节文件（模拟下载器非原子写留下的半成品）
    for t in COLS:
        if t == "balance":
            continue
        _write(s, t, "sh.600004", [_row(t, "sh.600004", "2020-04-10", "2019-12-31")])
    (s / "balance").mkdir(parents=True, exist_ok=True)
    (s / "balance" / "sh.600004.parquet").write_bytes(b"")

    return s


# ── 1. 列清单与列序：39 列，按表分块 ──────────────────────────
def test_columns_are_39_and_grouped_by_table(src, tmp_path):
    """列 = statDate + (每表: pubDate_<表> + 该表指标) + pubDate_max，共 39。

    硬编码期望值：规格规定的顺序，不 import 实现里的 TABLES。
    """
    build_events(out=tmp_path / "out", src=src)
    df = pd.read_parquet(tmp_path / "out" / "sh.600000.parquet")
    exp = ["statDate"]
    for t, cols in COLS.items():
        exp += [f"pubDate_{t}"] + cols
    exp += ["pubDate_max"]
    assert list(df.columns) == exp
    assert df.shape[1] == 39


# ── 2. 空串 → NaN（TRY_CAST 生效）────────────────────────────
def test_empty_string_becomes_nan(src, tmp_path):
    """baostock 落盘时缺失是空串不是 NULL；不 TRY_CAST 的话 isna() 恒为 0。"""
    build_events(out=tmp_path / "out", src=src)
    df = pd.read_parquet(tmp_path / "out" / "sh.600000.parquet").set_index("statDate")
    assert df.loc["2019-12-31", "currentRatio"] != df.loc["2019-12-31", "currentRatio"]  # NaN
    assert not pd.isna(df.loc["2020-03-31", "currentRatio"])
    assert df.currentRatio.dtype.kind == 'f'


# ── 3. FULL OUTER：缺一期时行仍在，那块全 NaN ────────────────
def test_missing_period_kept_and_nulled(src, tmp_path):
    """B 的 growth 缺 2020-03-31。INNER JOIN 会让这行消失 —— 那是丢样本。"""
    build_events(out=tmp_path / "out", src=src)
    df = pd.read_parquet(tmp_path / "out" / "sh.600018.parquet").set_index("statDate")
    assert len(df) == 3, f"B 应保留 3 期，实际 {len(df)}"
    assert pd.isna(df.loc["2020-03-31", "pubDate_growth"])
    assert pd.isna(df.loc["2020-03-31", "YOYNI"])
    assert not pd.isna(df.loc["2020-03-31", "roeAvg"]), "profit 那期有值，不该被连坐"


# ── 4. pubDate_max 取最晚（用 91 天夹具）──────────────────────
def test_pubdate_max_takes_latest(src, tmp_path):
    """真实数据实测：sh.600018 2019-12-31 快报 01-10 / 年报 04-10，差 91 天。

    取 least 会让这期在 01-10 就「齐全可见」= 前视 91 天。
    """
    build_events(out=tmp_path / "out", src=src)
    df = pd.read_parquet(tmp_path / "out" / "sh.600018.parquet").set_index("statDate")
    r = df.loc["2019-12-31"]
    assert r["pubDate_growth"] == pd.Timestamp("2020-01-10")
    assert r["pubDate_profit"] == pd.Timestamp("2020-04-10")
    assert r["pubDate_max"] == pd.Timestamp("2020-04-10"), "取 least 就会是 01-10"


# ── 5. 无 code 列、日期是 datetime ───────────────────────────
def test_no_code_column_and_datetime_dtype(src, tmp_path):
    """身份靠文件名；日期必须 datetime，否则下游 merge_asof 拿到字符串键。"""
    build_events(out=tmp_path / "out", src=src)
    df = pd.read_parquet(tmp_path / "out" / "sh.600000.parquet")
    assert "code" not in df.columns
    for c in df.columns:
        if c.startswith('pubDate') or c == 'statDate':
            assert str(df[c].dtype).startswith('datetime64'), c


# ── 6. 半写文件被跳过，不崩 ─────────────────────────────────
def test_zero_byte_parquet_is_skipped(src, tmp_path):
    """C 的 balance 是 0 字节（下载器非原子写的残留）。整批不该因此失败。"""
    meta = build_events(out=tmp_path / "out", src=src)
    assert meta["failures"] == []
    df = pd.read_parquet(tmp_path / "out" / "sh.600004.parquet")
    assert len(df) == 1
    assert pd.isna(df.loc[0, "currentRatio"]), "balance 整块缺失应为 NaN"
    assert not pd.isna(df.loc[0, "roeAvg"])