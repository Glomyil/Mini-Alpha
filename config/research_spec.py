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


def load_spec(since: int = 2016, until: str = "-", **override) -> RunSpec:
    """读一次配置并冻结成 RunSpec。override 仅供测试注入。"""
    vals = dict(
        since=since, until=until,
        fee_buy=FEE_BUY, fee_sell=FEE_SELL, slippage=SLIPPAGE, n_layers=N_LAYER,
        trading_days=TRADING_DAYS, periods_per_year=REBALANCE_PERIOD_PER_YEAR,
        init_cash=INIT_CASH,
        universe="all", limit_strict=False,
    )
    vals.update(override)
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