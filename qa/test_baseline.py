"""qa/test_baseline.py —— T-03 的规格（测试即规格）

每条描述一个【行为】，不是内部实现。
现在 ops/baseline.py 还不存在 ⇒ 全部会红在 ImportError ⇒ 那是预期的第一红 ✓
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 规格样本：一份 5 层的报告（层号前后有空格，列名打乱顺序以防写死下标）──
REPORT = """## 1. 分层指标

|层|期末净值|波动|年化|最大回撤|夏普|累计换手|费用合计|
|:--|:--|:--|:--|:--|:--|:--|:--|
| 0 | 0.9000 | 24.68% | -4.19% | -50.00% | 0.10 | 100.1725 | 0.0816 |
| 1 | 1.2000 | 21.00% | 6.00% | -40.00% | 0.30 | 101.3632 | 0.1390 |
| 2 | 1.3000 | 22.00% | 7.00% | -45.00% | 0.32 | 99.6513 | 0.1333 |
| 3 | 1.5000 | 23.00% | 8.00% | -50.00% | 0.35 | 102.2099 | 0.1337 |
| 4 | 1.6000 | 25.00% | 4.10% | -60.00% | 0.16 | 101.3631 | 0.0980 |
"""

PREV = REPORT.replace("-4.19%", "-3.81%").replace("24.68%", "23.51%").replace("4.10%", "3.71%")

LEDGER = """# BASELINE

## 变更记录（每张工单一行）

| 日期 | 工单 | 改了什么 | 数字变化 | 结论 |
|:--|:--|:--|:--|:--|
| 2026-10-17 | — | 建立基线 | — | 待改造 |
"""


def write(tmp_path, text, name="r.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ───────────────────────── 读报告 ─────────────────────────

def test_report_yields_all_five_layer_annualized_returns(tmp_path):
    """5 层的年化都要读出来（层1~3 不能丢）。"""
    from ops.baseline import read_report
    r = read_report(write(tmp_path, REPORT))
    assert r["ann"] == {"0": -4.19, "1": 6.00, "2": 7.00, "3": 8.00, "4": 4.10}


def test_turnover_is_kept_only_for_edge_layers(tmp_path):
    """换手只取层0 与层4。"""
    from ops.baseline import read_report
    r = read_report(write(tmp_path, REPORT))
    assert r["to"] == {"0": 100.1725, "4": 101.3631}


def test_alignment_row_is_not_read_as_a_layer(tmp_path):
    """|:--|:--| 是对齐行，不是数据行。"""
    from ops.baseline import read_report
    r = read_report(write(tmp_path, REPORT))
    assert "-" not in r["ann"]
    assert set(r["ann"]) == {"0", "1", "2", "3", "4"}


def test_columns_are_located_by_name_not_by_position(tmp_path):
    """列顺序被打乱时仍取对（样本里 波动 在 年化 之前）。"""
    from ops.baseline import read_report
    r = read_report(write(tmp_path, REPORT))
    assert r["vol"]["0"] == 24.68        # 第 3 列是波动
    assert r["ann"]["0"] == -4.19        # 第 4 列是年化


# ───────────────────────── 渲染变化 ─────────────────────────

def test_change_text_shows_previous_values_when_prev_exists(tmp_path):
    """有 _prev 时，每个数都带上前值。"""
    from ops.baseline import diff_text, read_report
    cur = read_report(write(tmp_path, REPORT, "c.md"))
    prev = read_report(write(tmp_path, PREV, "p.md"))
    out = diff_text(cur, prev)
    assert "前-3.81%" in out
    assert "前3.71%" in out
    assert "前100.1725" in out


def test_change_text_marks_dash_when_no_previous_report(tmp_path):
    """首次运行（没有 _prev）时标 (前-)，不许崩。"""
    from ops.baseline import diff_text, read_report
    out = diff_text(read_report(write(tmp_path, REPORT)), None)
    # 年化层0 / 年化层4 / 波动 / 换手层0 / 换手层4 / 单调性 = 6 项都该标 (前-)
    assert out.count("(前-)") == 6


def test_volatility_change_is_reported_as_relative_percent(tmp_path):
    """波动要显示 +4.98%（相对），不是 +1.17（绝对差）。"""
    from ops.baseline import diff_text, read_report
    cur = read_report(write(tmp_path, REPORT, "c.md"))
    prev = read_report(write(tmp_path, PREV, "p.md"))
    assert "+4.98%" in diff_text(cur, prev)


def test_flat_turnover_shows_no_change(tmp_path):
    """换手没变时应显示 +0.00。"""
    from ops.baseline import diff_text, read_report
    cur = read_report(write(tmp_path, REPORT, "c.md"))
    prev = read_report(write(tmp_path, PREV, "p.md"))
    out = diff_text(cur, prev)
    assert "100.1725(前100.1725, +0.00)" in out


# ───────────────────────── 写入台账 ─────────────────────────

def test_new_rows_land_after_the_last_data_row(tmp_path, monkeypatch):
    """新行必须排在最后一条数据行之后（不能插到表头下）。"""
    from ops import baseline
    ledger = tmp_path / "BASELINE.md"
    ledger.write_text(LEDGER, encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "report_x_close.md").write_text(REPORT, encoding="utf-8")

    monkeypatch.setattr(baseline, "BASELINE", ledger)
    monkeypatch.setattr(baseline, "REPORTS", reports)
    monkeypatch.setattr(sys, "argv", ["baseline", "--ticket", "T-03", "--change", "测试"])
    baseline.main()

    lines = ledger.read_text(encoding="utf-8").splitlines()
    old = [i for i, l in enumerate(lines) if "2026-10-17" in l][0]
    new = [i for i, l in enumerate(lines) if "T-03" in l][0]
    assert new == old + 1                  # 紧跟旧数据行
    assert "x 测试" in lines[new]