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

    # 2026-11-12（#11）：切片换成 slice_nav（两头都切）
    # 2026-11-13：切片抽进 build_factor_nav 之后，锚点改成「切片日志行」。
    #   守卫真正在乎的是【坏结果不该先打切片日志】，
    #   而日志现在位于哨兵之后 ⇒ 盯日志行才盯得住。
    assert re.search(r"MT\.slice_nav\(", src), "main.py 里找不到 MT.slice_nav( —— 净值切片被删了？"
    m = re.search(r"指标起算日", src)
    assert m, "main.py 里找不到 nav 切片日志行"

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


# ── 数据源契约（2026-11-18）：契约登记只能有一处，且两端必须对得上 ──────────

def test_kinds_is_the_single_registry():
    """loader.KINDS 是数据源的唯一登记表：两张源都要有 目录/时间列/必需列。"""
    from data import loader
    from data.cleaner import base

    assert set(loader.KINDS) == {"stock", "fundamental"}
    for kind, cfg in loader.KINDS.items():
        assert set(cfg) == {"store", "time_col", "required"}, kind
        assert cfg["time_col"] in ("date", "statDate"), kind
    # 用 is 而不是 ==：必需的是一份常量，不是两份相等的清单
    assert loader.KINDS["stock"]["required"] is base.REQUIRED_COLUMNS
    assert loader.KINDS["fundamental"]["required"] is base.FUNDAMENTAL_REQUIRED_COLUMNS

    # 两张表都以 kind 为键（loader 管「住哪」、base 管「长什么样」）。
    # 内容是同一批对象，但【键集合】必须对齐 —— 否则给一类数据源只登记了一半：
    #   只加在 loader  → base 查不到该 kind，align 那边会 ValueError
    #   只加在 base    → loader 查不到 store，当场 KeyError
    assert set(loader.KINDS) == set(base._REQUIRED_BY_KIND), (
        f"两张 kind 登记表的键不一致："
        f"loader={sorted(loader.KINDS)} vs base={sorted(base._REQUIRED_BY_KIND)}")


def test_fundamental_event_dir_has_one_source():
    """财务事件目录只在 config/constants.py 定义一次；merge 与 loader 都从那里取。"""
    from config import constants as C
    from data import fundamental_merge as FM
    from data import loader

    assert loader.KINDS["fundamental"]["store"] == C.FUNDAMENTAL_EVENT_DIR
    assert FM.EVENT_DIR == C.FUNDAMENTAL_EVENT_DIR, (
        f"merge 的 EVENT_DIR={FM.EVENT_DIR} 与 constants 不一致 —— "
        f"同一个目录定义了两遍（PITFALLS #26）")


def test_fundamental_contract_is_subset_of_merge_output():
    """财务必需列必须 ⊆ fundamental_merge 的真实产出列（两端对齐）。"""
    from data.cleaner import base
    from data.fundamental_merge import TABLES

    produced = {"statDate", "pubDate_max"}
    for t, cols in TABLES.items():
        produced |= {f"pubDate_{t}"} | set(cols)
    missing = [c for c in base.FUNDAMENTAL_REQUIRED_COLUMNS if c not in produced]
    assert not missing, f"财务契约里的 {missing} 不在 merge 产出列里"
    # 列数不写死：由「1 + 每表(1 个 pubDate + N 指标) + 1 个 pubDate_max」算出来，
    # 否则这里就成了 merge 规格的第二个真相源。
    expect = 1 + sum(1 + len(c) for c in TABLES.values()) + 1
    assert len(produced) == expect, f"merge 规格应是 {expect} 列，实际 {len(produced)}"

# ── 财务字段口径表（2026-11-23）──────────────────────────────

def test_every_fundamental_column_has_a_caliber_entry():
    """CALIBER 必须覆盖 TABLES 里全部 32 个指标列 —— 漏一个就红。

    为什么要有这条：32 列里混着【累计 / TTM / 同比 / 天数 / 时点存量】五种口径。
    不标出来，就会有人把 TTM 的和累计的放进同一个比值里 ——
    算出来的数没有金融含义，而且【不报错】。
    这和 PIPELINE_NAMES / _REQUIRED_BY_KIND 的对齐断言是同一个模式：
    第二份清单必须自带校验，否则迟早脱节（PITFALLS #26）。
    """
    from data.fundamental_merge import CALIBER, TABLES

    produced = {c for cols in TABLES.values() for c in cols}
    missing = sorted(produced - set(CALIBER))
    extra = sorted(set(CALIBER) - produced)
    assert not missing, f"这些列没有口径说明: {missing}"
    assert not extra, f"CALIBER 里有 TABLES 不存在的列: {extra}"
    assert len(produced) == 32, f"merge 规格应是 32 个指标，实际 {len(produced)}"
