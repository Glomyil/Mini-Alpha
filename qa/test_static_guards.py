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
    # 2026-11-10（T-6.2）：锚点从 "build_report" 收紧成 "RP.build_report("。
    # 原因：str.find 会命中【注释里的同名词】—— 中性化注释里提到 build_report，
    # 就骗过了守卫（实测 i_report 落到 3328 的注释上，真调用在 7554）。
    # 加 RP. 前缀和左括号之后，只有真调用能命中 ⇒ 这是【加强】不是放宽。
    i_report = src.find("RP.build_report(")
    # 2026-11-03：哨兵从「累计换手 < 20」改成「年化换手 < 0.7」，文案改过两次。
    # 所以锚点不再盯报错文案，改盯哨兵的【常量名】_MIN_TV_PER_YEAR ——
    # 文案随时会调，常量是这个哨兵的身份。
    i_sentinel = src.find("_MIN_TV_PER_YEAR")

    assert i_report > 0, "main.py 里找不到 build_report 调用"
    assert i_sentinel > 0, "找不到换手哨兵（_MIN_TV_PER_YEAR 常量）"
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


def _main_src() -> str:
    return (ROOT / "main.py").read_text(encoding="utf-8")


def _one(src: str, needle: str) -> int:
    """needle 必须【恰好出现一次】再返位置。

    为什么不用 str.find 直接找：注释里出现同名词会骗过守卫（2026-11-10 踩过，
    锚点从 build_report 收紧成 RP.build_report( 就是这个原因）。
    这里用「恰好一次」当契约：多一处就说明有人写了注释副本，或真的重复接线了。
    """
    n = src.count(needle)
    assert n == 1, f"main.py 里 {needle!r} 出现 {n} 次（应恰好 1 次）"
    return src.find(needle)


def test_main_applies_estimation_pool_before_neutralize():
    """点位1（estimation_pool）必须在 neutralize 之前 —— 它决定"用谁估 β"。"""
    src = _main_src()
    i_est = _one(src, "_fval = _apply_pool_mask(_fval, pool.estimation_pool)")
    i_neut = _one(src, "_fval = neutralize(")
    assert i_est < i_neut, "点位1 必须在 neutralize 之前"


def test_main_applies_invest_pool_between_neutralize_and_weights():
    """点位2（invest_pool）必须在 neutralize 之后、build_target_weights 之前。

    ⚠️ 这条守的是 2026-11-12 真踩过的 bug：点位1 接好了、点位2 漏了 ⇒
       universe_stage2 只改 cfg_hash、结果完全不变 ⇒ 口径声明与实际计算不一致。
    """
    src = _main_src()
    i_neut = _one(src, "_fval = neutralize(")
    i_inv = _one(src, "_fval = _apply_pool_mask(_fval, pool.invest_pool)")
    i_w = _one(src, "W, _ = build_target_weights(")
    assert i_neut < i_inv < i_w, "点位2 必须夹在 neutralize 与 build_target_weights 之间"


def test_main_passes_pool_into_weights_and_report():
    """三个下游必须吃 Pool：选股池、真 can_trade、基准池。"""
    src = _main_src()
    assert "select=pool.selectable()" in src, "weights 没收到 select=pool.selectable()"
    assert "pool.can_trade" in src, "weights 没收到 pool.can_trade（engine 侧的可交易性）"
    assert "bench_mask=pool.benchmark_mask()" in src, "build_report 没收到 bench_mask"
