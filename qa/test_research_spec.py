"""qa/test_research_spec.py —— research_spec 的规格（测试即契约）

守住三件事：
  ① 指纹的内容边界：9 个字段都在、顺序固定、用 | 分隔
  ② 内容的敏感度：改任一影响数字的字段 ⇒ hash 必变；改回来 ⇒ 回原值
  ③ 快照的副作用：只写 logs/config_snapshot/，内容与指纹一致

不测：具体 hash 值、banner 排版、yml 的 mtime 精确时间。
"""
import config.research_spec as R
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# 两个版本：只差 n_layers
SPEC_A = dict(since=2016, until="2026-09-07", n_layers=5,
              fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005)
SPEC_B = dict(SPEC_A, n_layers=10)

FIELDS = ["since", "until", "fee_buy", "fee_sell", "slippage",
          "n_layers", "trading_days", "periods_per_year", "init_cash"]


# ── T1 指纹的格式是契约 ────────────────────────────────────
def test_fingerprint_has_nine_ordered_fields():
    fp = R.load_spec(**SPEC_A).fingerprint()
    parts = fp.split("|")
    assert len(parts) == 9, "指纹应被 | 分成 9 段，实际 %r" % fp
    names = [p.split("=", 1)[0] for p in parts]
    assert names == FIELDS, "字段顺序变了：%r" % names


def test_fingerprint_values_are_not_glued_together():
    """少了逗号时相邻 f-string 会粘成一个串 —— 这条专门抓它。"""
    fp = R.load_spec(**SPEC_A).fingerprint()
    assert "since=2016until=" not in fp
    assert "since=2016|until=" in fp


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
