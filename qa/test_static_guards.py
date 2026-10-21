"""只读源码的静态守门测试（不需要任何数据 ⇒ CI 每次都跑）。

这里只放"结构不能坏"的检查：位置、顺序、来源链路、目录约定。
毫秒级，适合每次 push 都跑。
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_main_slices_nav_before_report():
    """main.py 必须有 nav.loc[rdates[0]:] 切片，且在【哨兵之后】【build_report 之前】。"""
    src = (ROOT / "main.py").read_text(encoding="utf-8")

    m = re.search(r"\.loc\[\s*rdates\[0\]\s*:\]", src)
    assert m, "main.py 里找不到 .loc[rdates[0]:] —— T-1.1 的切片被删了？"

    i_slice = m.start()
    i_report = src.find("build_report")
    i_sentinel = src.find("cumulative turnover < 20")

    assert i_report > 0, "main.py 里找不到 build_report 调用"
    assert i_sentinel > 0, "找不到哨兵（turnover < 20 的 raise）"
    assert i_sentinel < i_slice, "切片在哨兵之前 —— 坏结果会先被切片，日志会骗人"
    assert i_slice < i_report, "切片在 build_report 之后 —— 报告拿到的仍是全期 nav"


def test_report_start_comes_from_nav_index():
    """report.py 的 meta['start'] 必须取自 nav_df.index[0]，否则 main 的切片传不到报告区间。"""
    src = (ROOT / "backtest" / "report.py").read_text(encoding="utf-8")
    assert re.search(r'"start"\s*:\s*nav_df\.index\[0\]', src), (
        "report.py 的 meta['start'] 不再取自 nav_df.index[0] —— "
        "main 的切片将无法传达到报告区间")


def test_ci_ignores_heavy_dir():
    """守门：CI 必须显式 --ignore=qa/heavy，否则它会去读真面板（数据不在仓库 ⇒ 必红）。"""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--ignore=qa/heavy" in ci, (
        "ci.yml 里没有 --ignore=qa/heavy —— CI 会尝试收集需要真面板的测试")