from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parent

TRADING_DAYS = 244
REBALANCE_PERIOD_PER_YEAR = 12
INIT_CASH = 1.0

FEE_BUY = 0.00025
FEE_SELL = 0.00075
SLIPPAGE = 0.0005
N_LAYER = 5


@dataclass(frozen=True)
class RunSpec:
    since:     int
    until:     str
    split:     str
    freq:      str
    borrow_cost: float
    fee_buy:  float
    fee_sell:  float
    slippage:  float
    n_layers:  int
    trading_days: int
    periods_per_year: int
    init_cash: float
    universe: str
    limit_strict: bool
    cfg_hash: str
    cfg_mtime:  str

    def fingerprint(self) -> str:
        return "|".join([
            f"since={self.since}",
            f"until={self.until}",
            f"split={self.split}",
            f"freq={self.freq}",
            f"borrow_cost={self.borrow_cost}",
            f"fee_buy={self.fee_buy}",
            f"fee_sell={self.fee_sell}",
            f"slippage={self.slippage}",
            f"n_layers={self.n_layers}",
            f"trading_days={self.trading_days}",
            f"periods_per_year={self.periods_per_year}",
            f"init_cash={self.init_cash}",
        ])

    def banner(self) -> str:
        return (f"spec:{self.cfg_hash} 区间:{self.since}+ 到 {self.until} "
                f"费率:{self.fee_buy:.5f}/{self.fee_sell:.5f} 滑点:{self.slippage:.4f} "
                f"层数:{self.n_layers} 年化:{self.trading_days}d universe:{self.universe}")


def _yaml_mtime() -> str:
    p = CONFIG_DIR / "application.yml"
    if not p.exists():
        return "-"
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


# research 节允许出现的键。多一个就算配置写错了（见 _check_research_keys）
_RESEARCH_KEYS = frozenset({
    "since", "split", "fee_buy", "fee_sell", "slippage",
    "n_layers", "trading_days", "init_cash", "freq", "borrow_cost",
})
# 注意 periods_per_year 【不在此列】：它由 freq 推出，yml 里不该出现。
# 真写了 ⇒ _check_research_keys 会报「未知键」，强制你去删掉它而不是让它静默覆盖映射表。


# 调仓频率 → 每年调仓次数（单一源头是 freq，N 从这里推）
#   ME / QE 是精确值；W-FRI 用 52 近似（一年 52 周 + 1~2 天，实测 7 年 366 期 = 52.29/年）
#   ⚠️ 只用 pandas ≥ 2.2 的新别名：M / Q 已废弃，会刷 FutureWarning
_PERIODS_PER_YEAR = {"ME": 12, "QE": 4, "W-FRI": 52}


def _check_research_keys(config) -> None:
    """application.yml 的 research 节里出现未知键 => 立刻报错。

    为什么需要这条：_g() 在缺键时会静默回落代码默认，这本身是【设计】
    —— yml 只覆盖想改的那几项，其余用常量。但「键名拼错」也走同一条
    静默路径：把 split 写成 splti，你会得到一个「没有分段节」的报告，
    却以为是自己配成空的了。

    两者的唯一可检测信号：yml 里出现了不认识的键。
    缺键不报（正常），未知键报（拼错）。
    """
    raw = config.get("research") or {}
    unknown = set(raw) - _RESEARCH_KEYS
    if unknown:
        raise KeyError(
            f"application.yml 的 research 节有未知键：{sorted(unknown)}；"
            f"可用的是：{sorted(_RESEARCH_KEYS)}")


def load_spec(since: int | None = None, until: str = "-", **override) -> RunSpec:
    """读一次配置并冻结成 RunSpec。override 仅供测试注入。

    取值优先级：override > application.yml > 代码常量。

    since 的默认值必须是 None 而不是 2016 —— 否则调用方显式传了 2016，
    也会被 yml 的 research.since 静默盖掉；因为参数默认值本身分不出
    「调用方给了 2016」和「调用方没给」。
    """
    from config.config_manager import config

    _check_research_keys(config)

    def _g(key, default):
        v = config.get("research." + key, None)
        return default if v is None else v

    # 调用方显式给了 => 用它；否则看 yml；再否则用代码常量
    _since = since if since is not None else _g("since", 2016)

    vals = dict(
        since=_since,
        until=until,
        split=_g("split", ""),
        borrow_cost=float(_g("borrow_cost", 0.08)),
        fee_buy=_g("fee_buy", FEE_BUY),
        fee_sell=_g("fee_sell", FEE_SELL),
        slippage=_g("slippage", SLIPPAGE),
        n_layers=_g("n_layers", N_LAYER),
        trading_days=_g("trading_days", TRADING_DAYS),
        init_cash=_g("init_cash", INIT_CASH),
        universe="all",
        limit_strict=False,
    )
    vals.update(override)          # ★ 先合并 override，再推 freq / N

    # ★ freq 是唯一源头，N 由它推。两条硬约束：
    #   ① freq 和 periods_per_year 都【不在】上面的初始 vals 里、也【不从 yml 读】——
    #      yml 一旦有 periods_per_year，它会永远盖住映射表，freq 就不再是源头
    #      （PITFALLS #29「两个来源，一个永远赢」）。
    #   ② 必须在 vals.update(override) 【之后】算，否则 load_spec(freq="QE") 会得到
    #      freq=QE 但 periods_per_year=12（两个字段打架）。
    if "freq" in vals:                      # 只有 override 会把它放进来
        _freq = vals["freq"]
    else:
        _freq = _g("freq", "ME")
    if _freq not in _PERIODS_PER_YEAR:
        raise ValueError(f"未知调仓频率 {_freq!r}，可用：{sorted(_PERIODS_PER_YEAR)}")
    vals["freq"] = _freq
    # N 只认 override 显式给的，否则由 freq 推。用 is None 判，不用 or ——
    # or 会把 0 / "" 这类显式传值静默换成默认（PITFALLS #19 的精神：显式或无，没有"falsy"）。
    _n = vals.get("periods_per_year")
    vals["periods_per_year"] = int(_PERIODS_PER_YEAR[_freq] if _n is None else _n)
    mtime = _yaml_mtime()
    probe = RunSpec(**vals, cfg_hash="-", cfg_mtime=mtime)
    digest = hashlib.sha256(
        probe.fingerprint().encode("utf-8")).hexdigest()[:8]
    return RunSpec(**vals, cfg_hash=digest, cfg_mtime=mtime)


def save_snapshot(spec: RunSpec, when: datetime | None = None) -> Path:
    """把参数副本落到 logs/config_snapshot/<时间>_<hash>/。"""
    when = when or datetime.now()
    out = ROOT / "logs" / "config_snapshot" / \
        f"{when:%Y-%m-%d_%H%M}_{spec.cfg_hash}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.txt").write_text(spec.fingerprint() + "\n", encoding="utf-8")
    src = CONFIG_DIR / "application.yml"
    if src.exists():
        (out / "application.yml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out
