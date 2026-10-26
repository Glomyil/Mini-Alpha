"""qa/heavy/test_limits_real.py —— 涨跌停的端到端（要真 parquet，CI 跳过）

只测一条：真跑一只票，看 3 列是否产出、且抓到那次真实跌停。
数据不在就 skip —— 它依赖本机跑过清洗。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.needs_data

CLEAN = ROOT / "data" / "storage" / "cleaner_daily"
SAMPLE = CLEAN / "sh.600000.parquet"


@pytest.mark.skipif(not SAMPLE.exists(), reason="没有本机清洗产物，跳过端到端")
def test_real_stock_has_three_flag_columns():
    d = pd.read_parquet(SAMPLE)
    for col in ("flag_open_limit_up", "flag_open_limit_down", "flag_new_listing"):
        assert col in d.columns, f"缺列 {col}——limits 环节没跑？"
        assert str(d[col].dtype) == "int8", (col, d[col].dtype)


@pytest.mark.skipif(not SAMPLE.exists(), reason="没有本机清洗产物，跳过端到端")
def test_real_stock_catches_2020_02_03_limit_down():
    """2020-02-03 是疫情后首个交易日，千股跌停；浦发银行当天开盘就跌停（10.22）。"""
    d = pd.read_parquet(SAMPLE).sort_values("date").reset_index(drop=True)
    hit = d[d["flag_open_limit_down"] == 1]
    assert len(hit) >= 1, "十年里一天跌停都没有？flag 可能没生效"
    dates = {str(x)[:10] for x in hit["date"]}
    assert "2020-02-03" in dates, f"没抓到 2020-02-03（实际抓到：{sorted(dates)[:5]}）"


@pytest.mark.skipif(not SAMPLE.exists(), reason="没有本机清洗产物，跳过端到端")
def test_real_stock_ex_rights_days_are_not_limit_down():
    """2017-05-25（跌 16.42%）和 2016-06-23（跌 12.13%）是除权日，不许判成跌停。"""
    d = pd.read_parquet(SAMPLE).sort_values("date").reset_index(drop=True)
    for day in ("2017-05-25", "2016-06-23"):
        row = d[d["date"].astype(str).str[:10] == day]
        if len(row) == 0:
            continue
        assert int(row["flag_open_limit_down"].iloc[0]) == 0, f"{day} 是除权日，不该判跌停"
