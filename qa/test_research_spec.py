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
    fp = R.load_spec(**SPEC_A).fingerprint()
    parts = fp.split("|")
    assert len(parts) == 12, "指纹应被 | 分成 12 段，实际 %r" % fp
    names = [p.split("=", 1)[0] for p in parts]
    assert names == FIELDS, "字段顺序变了：%r" % names


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
