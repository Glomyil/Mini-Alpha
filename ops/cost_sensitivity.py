import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from backtest.assemble import rebalance_dates
from backtest.engine import nav_frame, run_all_layers
from backtest.factors import FACTORS
from backtest.metrics import (ann_return, equal_weight_nav, fee_total,
                              turnover_total)
from backtest.panel import load_panels
from backtest.pipeline import run_factor

SLIPS = (0.0005, 0.002, 0.005)
OUT = Path(__file__).resolve().parents[1] / "reports" / "cost_sensitivity.md"


def main():
    panels = load_panels()
    rdates = [d for d in rebalance_dates(panels["hfq_close"].index) if d.year >= 2016]

    bench = equal_weight_nav(panels).reindex(panels["hfq_close"].index).ffill()
    bench = bench.loc[rdates[0]:]
    bench_ann = ann_return(bench)
    print("基准(全A等权, 不扣费)年化 {:.2%}".format(bench_ann))

    rows = []
    for name in FACTORS:
        print("  {} ...".format(name), end=" ", flush=True)
        _, W, res0 = run_factor(name, panels, rdates, slippage=SLIPS[0])

        by_slip = {SLIPS[0]: res0}
        for sl in SLIPS[1:]:
            by_slip[sl] = run_all_layers(W, panels, rdates, slippage=sl)

        for sl, res in by_slip.items():
            nav = nav_frame(res).loc[rdates[0]:]
            ann4 = ann_return(nav[4])
            rows.append({
                "slip": sl,
                "factor": name,
                "ann4": ann4,
                "excess": ann4 - bench_ann,
                "turnover": turnover_total(res[4]["turnover"]),
                "fees": fee_total(res[4]["fees"]),
            })
        print("ok")

    df = pd.DataFrame(rows)
    df["rank_ann"] = df.groupby("slip")["ann4"].rank(
        ascending=False, method="min").astype(int)
    df["rank_exc"] = df.groupby("slip")["excess"].rank(
        ascending=False, method="min").astype(int)

    for sl in SLIPS:
        sub = df[df["slip"] == sl].sort_values("ann4", ascending=False)
        print("")
        print("-- 滑点 {:.0f}bp --".format(sl * 1e4))
        for _, row in sub.iterrows():
            print("  {:>2}. {:<14} 年化 {:>7.2%}  超额 {:>+7.2%}  换手 {:>6.2f}  费用 {:>6.3f}".format(
                int(row["rank_ann"]), row["factor"], row["ann4"],
                row["excess"], row["turnover"], row["fees"]))

    lines = ["# 成本三档敏感性（T-3.2）", "",
             "> 只改滑点，佣金固定 0.00025/0.00075。基准（全A等权）不扣费，三档相同。", "",
             "> 基准年化 **{:.2%}**".format(bench_ann), "",
             "## 1. 三档总表", "",
             "| 滑点 | 排名(年化) | 排名(超额) | 因子 | 层4年化 | 超额 | 累计换手 | 费用合计 |",
             "| :-- | --: | --: | :-- | --: | --: | --: | --: |"]
    for sl in SLIPS:
        sub = df[df["slip"] == sl].sort_values("ann4", ascending=False)
        for _, row in sub.iterrows():
            lines.append("| {:.0f}bp | {} | {} | `{}` | {:.2%} | {:+.2f}% | {:.2f} | {:.4f} |".format(
                row["slip"] * 1e4, int(row["rank_ann"]), int(row["rank_exc"]),
                row["factor"], row["ann4"], row["excess"] * 100,
                row["turnover"], row["fees"]))

    lo = df[df["slip"] == SLIPS[0]].set_index("factor")["rank_ann"]
    hi = df[df["slip"] == SLIPS[-1]].set_index("factor")["rank_ann"]
    lines += ["", "## 2. 排序变化（5bp → 50bp）", "",
              "| 因子 | 5bp 排名 | 50bp 排名 | Δrank | 判断 |",
              "| :-- | --: | --: | --: | :-- |"]
    for f in lo.sort_values().index:
        d = int(hi[f]) - int(lo[f])
        verdict = "稳健" if d == 0 else ("略变" if abs(d) == 1 else "敏感")
        lines.append("| `{}` | {} | {} | {:+d} | {} |".format(
            f, int(lo[f]), int(hi[f]), d, verdict))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("")
    print("已写 {}".format(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
