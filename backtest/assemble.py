from __future__ import annotations
import sys
import time
import pandas as pd
from backtest.panel import load_panels
from backtest.features import FEATURES
from backtest.factors import FACTORS


def build_features(panels, factor_names, cache=None):
    cache = {} if cache is None else cache
    out = {}
    for name in factor_names:  # 这里获取需要计算什么因子的指标
        spec = FACTORS[name]
        missing = [t for t in spec.get(
            "requires", []) if t not in panels]  # 获取缺失值
        if missing:
            print(f"  跳过 {name}：面板里没有 {missing}")  # 打印缺失部分
            continue
        for feat_name, params in spec["needs"]:  # 读取factor中因子所需指标和对应参数
            key = (feat_name, tuple(sorted(params.items())))
            if key not in cache:  # 同一个指标只计算一次，谁要谁拿走
                cache[key] = FEATURES[feat_name](panels, **params)
            out[feat_name + "_" +
                "_".join(str(v) for _, v in sorted(params.items()))] = cache[key]
    return out


def rebalance_dates(index, rule="ME"):  # 确认好调仓频率，这个写入的代表频率是一月一次
    return pd.Series(index, index=index).resample(rule).last().dropna().tolist()


def factor_corr(feats, factor_names, dates, min_names=30):  # 注：实际上进行多因子同时回测的时候为了防止把多个因子当作一个来看
    """相关性小表：**定向后**的因子，每个调仓日算一次截面 Spearman，再对期数取平均"""
    oriented = {}
    skipped = []
    for name in factor_names:
        spec = FACTORS[name]
        try:
            val = spec["fn"](feats, {})
        except KeyError:               # 它 needs 的指标没被算出来（例如 requires 缺表 → 被跳过）
            continue
        oriented[name] = val * spec["sign"]        # 方向只在这里用一次
    mats = []
    for dt in dates:
        snap = {}
        for name, s in oriented.items():
            if dt in s.index:
                v = s.loc[dt].dropna()
                if len(v) >= min_names:
                    # 先 rank → 再 pearson 等价于 Spearman
                    snap[name] = v.rank()
        if len(snap) < len(oriented):            # 这一天有因子缺值 → **整天丢掉**，否则整行会被 NaN 污染
            skipped.append(str(dt.date()))
            continue
        if len(snap) >= 2:
            mats.append(pd.DataFrame(snap).corr(method="pearson"))
    if skipped:
        print(f"  相关性小表：跳过 {len(skipped)} 个调仓日（有因子当天有效值不足）：{skipped[:3]}")
    return sum(mats) / len(mats) if mats else None


def main():
    FACTOR_LIST = sys.argv[1:] or list(FACTORS)
    t0 = time.time()
    panels = load_panels()
    print(f"面板 {panels['hfq_close'].shape}   用时 {time.time() - t0:.1f}s")
    cache = {}
    t1 = time.time()
    feats = build_features(panels, FACTOR_LIST, cache)
    print(f"按需装配：{len(FACTOR_LIST)} 个因子声明 → {len(feats)} 张指标表（缓存 {len(cache)} 条，用时 {time.time() - t1:.2f}s）")
    print()
    print(f"{'指标表':<12}{'shape':<16}{'NaN 占比':>10}")
    for k, v in feats.items():
        print(f"{k:<12}{str(v.shape):<16}{float(v.isna().to_numpy().mean()):>9.2%}")
    if len(FACTOR_LIST) == 0:                      # 清单是空的：什么都没算
        print()
        print("（清单是空的 → 一个因子都没跑；把 FACTOR_LIST 或命令行参数写上因子名）")
    elif len(FACTOR_LIST) == 1:                    # 只测一个：相关性小表没有意义（只有 1 列）
        print()
        print(f"（只跑了 1 个因子 {FACTOR_LIST[0]} → 相关性小表跳过：它需要 ≥2 个因子才有意义）")
    else:
        dates = [d for d in rebalance_dates(
            panels["hfq_close"].index) if d.year >= 2016]
        corr = factor_corr(feats, FACTOR_LIST, dates)
        print()
        print(f"相关性小表（定向后，{len(dates)} 个调仓日平均；|ρ|>0.7 = 基本同一个因子）")
        print(corr.round(2).to_string())
        hot = [(a, b, round(float(corr.loc[a, b]), 2))
               for i, a in enumerate(corr.index) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.7]
        print("|ρ|>0.7 的组合：" + (str(hot) if hot else "无"))
    print()
    print("下一步（engine 那一格）：因子值 → 分层权重 → T+1 开盘撮合 → 净值 / 指标 / 报告")


if __name__ == "__main__":
    main()
