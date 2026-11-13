"""qa/test_research_spec.py —— research_spec 的规格（测试即契约）

守住五件事：
  ① 指纹的内容边界：10 个字段都在、顺序固定、用 | 分隔
  ② 内容的敏感度：改任一影响数字的字段 ⇒ hash 必变；改回来 ⇒ 回原值
  ③ 快照的副作用：只写 logs/config_snapshot/，内容与指纹一致
  ④ yml 的 research 节有未知键 ⇒ KeyError（拼错不许静默回退）
  ⑤ 调用方显式传 since ⇒ 不许被 yml 覆盖（优先级 override > yml > 常量）

不测：具体 hash 值、banner 排版、yml 的 mtime 精确时间。
"""
import config.research_spec as R
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# 两个版本：只差 n_layers
# split 显式传值：application.yml 里现在写着 "2020-08-31"，不传就会被它盖住，
# 那样测试的结果取决于配置文件 —— 不是测试该有的样子。
SPEC_A = dict(since=2016, until="2026-09-07", split="", n_layers=5,
              fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005)
SPEC_B = dict(SPEC_A, n_layers=10)

# 2026-11-02 T-3.3：加 split（放在 until 后面，语义相邻）⇒ 9 段变 10 段
# 2026-11-03 T-3.4：加 freq（第 4 位）和 borrow_cost（第 5 位）⇒ 10 段变 12 段。
# 位置和 RunSpec.fingerprint() 里的顺序一一对应；改任一侧这条都会红。
FIELDS = ["since", "until", "split", "freq", "borrow_cost",
          "fee_buy", "fee_sell", "slippage",
          "n_layers", "trading_days", "periods_per_year", "init_cash"]


# ── T1 指纹的格式是契约 ────────────────────────────────────
def test_fingerprint_has_twelve_ordered_fields():
    # 2026-11-10（T-6.2）：enabled 现在会被 yml 打开（跑对照实验就要切），
    # 所以两态都必须显式传 —— 否则测试结果取决于配置文件里的那个开关，
    # 与本文件 SPEC_A 注释的原则一致：「不传就会被 yml 盖住，不是测试该有的样子」。
    # 2026-11-13：cut 非空会追加一段 ⇒ 这里显式置空，测的是【基础 12 段】的顺序
    fp = R.load_spec(**SPEC_A, neutralize=False, include_st=True, cut="").fingerprint()
    parts = fp.split("|")
    assert len(parts) == 12, "关闭中性化时指纹应 12 段，实际 %r" % fp
    names = [p.split("=", 1)[0] for p in parts]
    assert names == FIELDS, "字段顺序变了：%r" % names

    # 开启时 = 12 + 5 段；且【前 12 段一个字都不能变】——
    # 这是"历史台账继续可比"的唯一保证，动了就等于所有旧 cfg_hash 失效。
    parts_on = R.load_spec(**SPEC_A, neutralize=True, include_st=True, cut="").fingerprint().split("|")
    assert len(parts_on) == 17, "开启中性化时应 17 段，实际 %r" % parts_on
    assert parts_on[:12] == parts, "开启中性化不许动前 12 段（否则历史 cfg_hash 全失效）"


def test_fingerprint_values_are_not_glued_together():
    """少了逗号时相邻 f-string 会粘成一个串 —— 这条专门抓它。"""
    fp = R.load_spec(**SPEC_A).fingerprint()
    assert "since=2016until=" not in fp
    assert "since=2016|until=" in fp
    assert "until=2026-09-07|split=" in fp or "until=2026-09-07split=" not in fp


# ── T2 同一版本多次算，hash 必须完全相同 ────────────────────
def test_same_spec_yields_same_hash_every_time():
    hashes = [R.load_spec(**SPEC_A).cfg_hash for _ in range(3)]
    assert len(set(hashes)) == 1, "同一版本算出多个 hash：%r" % hashes
    assert len(hashes[0]) == 8


# ── T3 改一个字段，hash 必须变 ─────────────────────────────
def test_changing_one_field_changes_hash():
    assert R.load_spec(**SPEC_A).cfg_hash != R.load_spec(**SPEC_B).cfg_hash


# ── T4 改回来要回到原值（内容寻址）──────────────────────────
def test_hash_returns_to_original_when_field_restored():
    first = R.load_spec(**SPEC_A).cfg_hash
    _ = R.load_spec(**SPEC_B).cfg_hash
    again = R.load_spec(**SPEC_A).cfg_hash
    assert first == again


# ── T5 钉住年化常数 ───────────────────────────────────────
def test_trading_days_constant_is_244():
    assert R.TRADING_DAYS == 244, "A 股年化用 244；252 是美股的数"


# ── T6 快照只写到指定目录，且内容与指纹一致 ─────────────────
def test_snapshot_writes_fingerprint_and_yaml_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    spec = R.load_spec(**SPEC_A)
    out = R.save_snapshot(spec)

    assert out.parent == tmp_path / "logs" / "config_snapshot"
    assert out.name.endswith(spec.cfg_hash)
    assert (out / "spec.txt").read_text(encoding="utf-8").strip() == spec.fingerprint()
    assert (out / "application.yml").exists()

# ── T7 改 split，hash 必须变（加这个字段的唯一目的）─────
def test_changing_split_changes_hash():
    """split 是口径的一部分：同一份数据切在不同日期 => 必须能被台账区分。

    如果 split 没进 fingerprint，两个不同分界的报告会算出同一个 cfg_hash，
    台账里两行分不清「这份用的是哪个分界」—— 那就等于不知道自己在看什么。
    """
    a = R.load_spec(**dict(SPEC_A, split="2020-08-31")).cfg_hash
    b = R.load_spec(**dict(SPEC_A, split="2021-06-30")).cfg_hash
    c = R.load_spec(**dict(SPEC_A, split="")).cfg_hash

    assert a != b, "换分界点 hash 没变 => split 没进 fingerprint"
    assert a != c, "有分界 vs 无分界 hash 没变"
    assert len({a, b, c}) == 3

# ── T8 未知键必须报错（拼错不能被静默回退）─────────────
def test_unknown_research_key_raises(monkeypatch):
    """application.yml 的 research 节里出现不认识的键 => KeyError。

    为什么要有这条：_g() 缺键时会静默回落代码默认（这是设计 —— yml 只覆盖
    想改的那几项）。但「键名拼错」走的是同一条静默路径：把 split 写成 splti，
    你会得到一份「没有分段节」的报告，却以为是自己配成空的了。

    缺键不报（正常），未知键报（拼错）—— 这是两者唯一的可检测区别。
    """
    import config.research_spec as R
    from config.config_manager import config

    raw = config.get("research") or {}
    monkeypatch.setattr(config, "_config",
                        {**config._config, "research": {**raw, "splti": "2020-08-31"}})

    with pytest.raises(KeyError, match="未知键"):
        R.load_spec()


# ── T9 调用方显式传 since => 不许被 yml 覆盖 ─────────────
def test_caller_since_beats_yml(monkeypatch):
    """优先级：override > application.yml > 代码常量。

    这条守的是 since 的默认值必须是 None 而不是 2016 ——
    否则调用方显式传 2016 时，参数默认值分不出「给了」和「没给」，
    yml 的 research.since 就会把它静默盖掉。
    """
    import config.research_spec as R
    from config.config_manager import config

    raw = config.get("research") or {}
    monkeypatch.setattr(config, "_config",
                        {**config._config, "research": {**raw, "since": 2020}})

    assert R.load_spec().since == 2020, "yml 有值时应当用 yml"
    assert R.load_spec(since=2018).since == 2018, "调用方显式给了 => 不许被 yml 盖掉"
    assert R.load_spec(since=2016).since == 2016, "传 2016 也要生效（不是'用默认'）"


# ── T7 宇宙两字段（2026-11-12 §5.2）─────────────────────────
# 设计：estimation_pool（点位1）/ invest_pool（点位2），代码里叫 stage1/stage2。
# 关键不变式：est 一旦收窄，invest 就【不可能宽于它】（因子值已 NaN ⇒ 传不动）
#   ⇒ est=hs300 时，invest=all 与 invest=hs300 是同一件事，hash 必须相同。
def test_pool_default_keeps_hash_unchanged():
    """默认 all/all ⇒ 指纹一个字都不加（历史 hash 继续可比）"""
    a = R.load_spec(**SPEC_A, universe_stage1="all", universe_stage2="all")
    b = R.load_spec(**SPEC_A)
    assert a.cfg_hash == b.cfg_hash
    assert "est_pool" not in a.fingerprint()
    assert "inv_pool" not in a.fingerprint()


def test_pool_invest_only_appends_one_segment():
    base = len(R.load_spec(**SPEC_A).fingerprint().split("|"))
    s = R.load_spec(**SPEC_A, universe_stage1="all", universe_stage2="hs300")
    assert len(s.fingerprint().split("|")) == base + 1
    assert "inv_pool=hs300" in s.fingerprint()


def test_pool_estimation_appends_and_normalizes_invest():
    """est 收窄后：invest=all 与 invest=est 结果等价 ⇒ 必须同一个 hash"""
    base = len(R.load_spec(**SPEC_A).fingerprint().split("|"))
    a = R.load_spec(**SPEC_A, universe_stage1="hs300", universe_stage2="all")
    b = R.load_spec(**SPEC_A, universe_stage1="hs300", universe_stage2="hs300")
    assert len(a.fingerprint().split("|")) == base + 1
    assert "est_pool=hs300" in a.fingerprint()
    assert a.cfg_hash == b.cfg_hash, "est=hs300 时 inv=all 与 inv=hs300 是同一件事"


def test_pool_narrower_invest_appends_two_segments():
    base = len(R.load_spec(**SPEC_A).fingerprint().split("|"))
    a = R.load_spec(**SPEC_A, universe_stage1="hs300", universe_stage2="all")
    c = R.load_spec(**SPEC_A, universe_stage1="hs300", universe_stage2="zz500")
    assert len(c.fingerprint().split("|")) == base + 2
    assert c.cfg_hash != a.cfg_hash


def test_pool_rejects_unknown_universe():
    with pytest.raises((ValueError, KeyError)):
        R.load_spec(**SPEC_A, universe_stage1="hs999")
    with pytest.raises((ValueError, KeyError)):
        R.load_spec(**SPEC_A, universe_stage2="zz1000")


def test_pool_allowlist_matches_data_module():
    """config 不 import data（避免反向依赖），靠这条测试保证两份白名单不分叉"""
    import data.index_snapshot as D
    # "all" 是 config 侧的哨兵（不读快照），不在 loader 的 INDEX_CODES 里；
    # 剩下四个真实指数必须一一对应。
    assert tuple(x for x in R.INDEX_UNIVERSES if x != "all") == tuple(D.INDEX_CODES)


# ── T8 until 的空值语义（2026-11-12 #12）──────────────────
# 现状：main.py 把面板末日塞进 until ⇒ 面板每更新一天，cfg_hash 就变
# ⇒ 一个参数都没改，hash 却变了，台账永远对不齐。
# 修法：未指定时记 "auto"（不依赖数据），显式指定才记具体日期。


# SPEC_A 里已经有 until="2026-09-07"，再传一次会 TypeError ⇒ 先剔掉
_NO_UNTIL = {k: v for k, v in SPEC_A.items() if k != "until"}


def test_until_defaults_to_auto_and_is_data_independent():
    a = R.load_spec(**_NO_UNTIL, until=None)
    b = R.load_spec(**_NO_UNTIL, until="auto")
    assert a.until == "auto", f"\u672a\u6307\u5b9a until \u21d2 \u5e94\u4e3a auto\uff0c\u5b9e\u6d4b {a.until!r}"
    assert a.cfg_hash == b.cfg_hash, "until=None 与 until='auto' 必须同 hash"


def test_until_explicit_changes_hash():
    a = R.load_spec(**_NO_UNTIL, until=None)
    c = R.load_spec(**_NO_UNTIL, until="2023-10-31")
    assert a.cfg_hash != c.cfg_hash, "显式指定 until ⇒ hash 必须不同"
    assert "until=2023-10-31" in c.fingerprint()


def test_until_does_not_change_segment_count():
    # 不传就会被 yml 盖住（neutralize/include_st 会改段数）⇒ 显式传
    assert len(R.load_spec(**_NO_UNTIL, until=None, neutralize=False, include_st=True, cut="").fingerprint().split("|")) == 12
    assert len(R.load_spec(**_NO_UNTIL, until="2023-10-31", neutralize=False, include_st=True, cut="").fingerprint().split("|")) == 12


def test_until_rejects_date_before_since():
    with pytest.raises((ValueError, KeyError)):
        R.load_spec(**_NO_UNTIL, until="2010-01-01")


# ── T9 cut（2026-11-13）：样本集分界日 ─────────────────────────────
# 与 split 的区别：split 只切【报告分段表】，cut 是【真实样本集分界】，
# 且它进指纹 —— 因为它决定了「这次评价用的是哪两个时间段」。


def test_cut_enters_fingerprint_only_when_set():
    base = dict(neutralize=False, include_st=True)
    a = R.load_spec(**base, cut="")
    b = R.load_spec(**base, cut="2020-01-02")
    assert len(a.fingerprint().split("|")) == 12, "空 cut 不许改段数"
    assert len(b.fingerprint().split("|")) == 13, "非空 cut 追加一段"
    assert b.fingerprint().split("|")[:12] == a.fingerprint().split("|"), "前 12 段不许变"
    assert b.fingerprint().split("|")[-1] == "cut=2020-01-02"
    assert a.cfg_hash != b.cfg_hash, "不同切点必须是不同配置"


def test_until_can_reference_cut():
    a = R.load_spec(**_NO_UNTIL, until="@cut", cut="2020-01-02")
    b = R.load_spec(**_NO_UNTIL, until="2020-01-02", cut="2020-01-02")
    assert a.until == "2020-01-02", "@cut 至少要能解析成日期"
    assert a.cfg_hash == b.cfg_hash, "引用与直写必须同 hash（否则同一件事两个台账）"


def test_cut_validated():
    with pytest.raises(ValueError):
        R.load_spec(**_NO_UNTIL, cut="2010-01-01")      # 早于 since
    with pytest.raises(ValueError):
        R.load_spec(**_NO_UNTIL, until="@cut", cut="")   # 引用空值
    with pytest.raises(ValueError):
        R.load_spec(**_NO_UNTIL, cut="2020/01/02")      # 格式不对
