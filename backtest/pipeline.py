from __future__ import annotations
from backtest.assemble import build_features
from backtest.engine import run_all_layers
from backtest.factors import FACTORS
from backtest.weights import build_target_weights


def run_factor(name, panels, rdates, **engine_kw):
    if name not in FACTORS:
        raise KeyError(f"未知因子 {name!r}；FACTORS 里有: {sorted(FACTORS)}")
    spec = FACTORS[name]
    missing = [t for t in spec.get("requires", []) if t not in panels]
    if missing:
        raise KeyError(
            f"因子 {name} 需要面板里没有的字段: {missing}；"
            f"现有字段: {sorted(panels)}")

    fval = spec["fn"](build_features(panels, [name]), {}) * spec["sign"]
    W, _ = build_target_weights(fval, rdates, panels["can_trade"])
    return fval, W, run_all_layers(W, panels, rdates, **engine_kw)
