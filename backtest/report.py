from __future__ import annotations
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from backtest import metrics as MT
from pathlib import Path
from loguru import logger
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "reports"


def md_table(df: pd.DataFrame, index_name: str = "层",
             pct_cols=("年化", "波动", "最大回撤", "基准年化", "超额")) -> str:
    cols = [index_name]+[str(c)for c in df.columns]
    head = "|"+"|".join(cols)+"|"
    sep = "|"+"|".join(":--"for _ in cols)+"|"
    body = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2%}" if c in pct_cols else f"{v:.4f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def banner(meta: dict) -> str:
    """口径声明区块 —— 报告的签名（口径②）；缺一项就算没做这一格"""
    return "\n".join([
        "> **口径声明**",
        f"> - 因子：`{meta['factor']}` ｜ 区间：{meta['start']} ~ {meta['end']} ｜ 调仓：月频（{meta['n_rdate']} 期）",
        f"> - 净值口径：**{meta['val_basis']}**（收盘 / 开盘两套，另一套作诊断）｜ 成交：**T+1 开盘**（缺价即不成交）",
        f"> - 年化常数：**{meta['trading_days']}** 交易日/年（A 股；252 是美股）｜ IC-IR 用 {meta['periods_per_year']} 期/年",
        f"> - 成本：佣金 {meta['fee_buy']}/{meta['fee_sell']} ｜ 滑点 {meta['slippage']}（单边）｜ 无融券费 / 无整手",
        f"> - 多空：比值法（主）· 日差法 · 月差法 —— 结论以**比值法**为准",
        f"> - 生成时间：{meta['stamp']}",
    ])


def nav_plot(nav_df: pd.DataFrame, path: Path, *, title: str = "分层净值曲线",
             bench: pd.Series | None = None) -> Path:
    """画 5 层净值 + 可选基准线（只画真实点，不 ffill；只 savefig，不 show）。

    2026-10-30（T-3.1）三处修改，都是为了让图真的能读：
      · 基准线要【先归一化到 1.0】：它从 2015 年累乘起，直接叠到 2016 起的 nav 上
        会从 1.6 开始 —— 表里的年化没错，但图上起点不同等于骗人。
      · y 轴改【对数】：净值跨 0.5~4.0 时，线性轴会把中间几层压成一团看不清。
      · 多空（层4净值 / 层0净值）是【比值】不是【净值】，必须挪到右轴，
        否则它跳到 3.5 会把左轴的 5 层全部压扁（旧版的图就是这个毛病）。
    """
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    for L in nav_df.columns:
        ax.plot(nav_df.index, nav_df[L].to_numpy(), lw=1.2, alpha=0.85, label=f"层 {L}")
    if bench is not None:
        ax.plot(bench.index, bench.to_numpy(), lw=2.4, color="#2c7fb8",
                label="基准(全A等权)")
    ax.axhline(1.0, color="#9aa5b1", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("净值（对数轴，起点 = 1.0）")

    ls = nav_df[nav_df.columns[-1]] / nav_df[nav_df.columns[0]]
    ax2 = ax.twinx()
    ax2.plot(nav_df.index, ls.to_numpy(), lw=1.2, ls="--", color="#c0392b",
             alpha=0.55, label="多空(比值法)")
    ax2.set_ylabel("多空比值", fontsize=8, color="#c0392b")
    ax2.tick_params(labelsize=7, colors="#c0392b")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=4, fontsize=8, frameon=False, loc="upper left")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report(factor, nav_df, panels, rdates, res=None, *, rf=0.0, outdir: Path = OUT,
                 val_basis: str = "close", note: str = "", factor_val=None) -> dict:
    layer_tab = MT.layer_table(nav_df, res, rf=rf)

    # ── T-3.1 基准线：全 A 等权（每日再平衡、未扣费，基准偏高 => 超额偏保守）──
    _bench = None
    if panels is not None and "ret_1d_hfq" in panels and "can_trade" in panels:
        _bench = MT.equal_weight_nav(panels).reindex(nav_df.index).ffill()
        # 归一化到 1.0：它从 2015 年累乘起，直接叠到 2016 起的 nav 上会从 1.6 开始。
        # 年化不受影响（ann_return 是相对算的），但图上起点不一致等于骗人 ✗
        _bench = _bench / float(_bench.iloc[0])
        _bann = MT.ann_return(_bench)
        layer_tab["基准年化"] = _bann
        layer_tab["超额"] = layer_tab["年化"] - _bann
    else:
        layer_tab["基准年化"] = "缺基准数据"
        layer_tab["超额"] = "缺基准数据"

    ls = MT.ls_summary(nav_df, rdates)

    ic = ic_sum = None
    if panels is not None and factor is not None:
        # not provided -> compute it once (legacy path)
        if factor_val is None:
            from backtest.assemble import build_features
            from backtest.factors import FACTORS
            spec = FACTORS[factor]
            factor_val = spec["fn"](build_features(
                panels, [factor]), {}) * spec["sign"]
        ic = MT.rank_ic(factor_val, panels, rdates)
        ic_sum = MT.ic_summary(ic)
    stamp = pd.Timestamp.today().strftime(
        "%Y-%m-%d")          # 只进报告正文（文件名不带日期 → 每次运行覆盖同一份）
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"nav_{factor}_{val_basis}.png"
    md = outdir / f"report_{factor}_{val_basis}.md"
    if md.exists():                                     # 覆盖前留一份"上一次"，供前后比对（每因子只留一版）
        for _p in (md, png):
            if _p.exists():
                shutil.copy2(_p, _p.with_name(_p.stem + "_prev" + _p.suffix))
    nav_plot(nav_df, png, title=f"{factor} 分层净值", bench=_bench)
    meta = {"factor": factor, "start": nav_df.index[0].date(), "end": nav_df.index[-1].date(),
            "n_rdate": len(rdates), "trading_days": MT.TRADING_DAYS,
            "periods_per_year": MT.REBALANCE_PERIODS_PER_YEAR, "val_basis": val_basis,
            "fee_buy": 0.00025, "fee_sell": 0.00075, "slippage": 0.0005, "stamp": stamp}

    parts = [f"# 因子回测报告 · `{factor}`", "", banner(meta), ""]

    # ── 成交失败（T-2.4）：放在首屏，在「## 1. 分层指标」之前 ──
    # ⚠️ res 是 list[dict]（每层一个账本，来自 run_all_layers），不是单个 dict。
    #    报告用【最高层】（res[-1]）的账本来报 —— 和 ROADMAP 里「层4 掉多少 pp」对得上 ✓
    _top = res[-1] if isinstance(res, (list, tuple)) and len(res) > 0 else None
    if _top is not None and panels is not None and "blocked_buy" in _top:
        ex = _top["blocked_buy"].index
        n_ex = len(ex)
        bb, bs = _top["blocked_buy"], _top["blocked_sell"]
        r = panels["reason"].reindex(ex) if "reason" in panels else None

        n_buy = int((bb > 0).to_numpy().sum())
        n_lu = int(((bb > 0) & (r == 2)).to_numpy().sum()
                   ) if r is not None else 0
        n_ld = int(((bs > 0) & (r == 3)).to_numpy().sum()
                   ) if r is not None else 0

        pp = MT._blocked_pp(_top, panels)
        parts += [
            "## 0. 成交失败（P2 核心结论）", "",
            f"- 目标买入里，**{n_lu} 笔撞上涨停**没买成（共 {n_ex} 期）",
            f"- 目标卖出里，**{n_ld} 笔撞上跌停**没卖成",
            f"- 另有 **{n_buy - n_lu} 笔**因停牌 / 不在市 / 无价没买成",
            f"- 这些没成交的单子，一年大约 **{pp:.2f} pp** 收不到",
            "",
        ]

    parts += ["## 1. 分层指标", "", md_table(layer_tab), ""]
    if ic_sum is not None:
        logger.info(f"ic 有效:{ic_sum['n']} 无效:{ic_sum.get('n_bad', 0)} "
                    f"mean:{ic_sum['ic_mean']:.4f} ir:{ic_sum['ic_ir']:.4f}")
        parts += ["## 2. IC（标签 = 下一个调仓日的区间收益）", "",
                  f"- IC 均值 **{ic_sum['ic_mean']:.4f}** ｜ 标准差 {ic_sum['ic_std']:.4f} "
                  f"｜ **IC-IR {ic_sum['ic_ir']:.4f}** ｜ 期数 {ic_sum['n']}", ""]
    parts += ["## 3. 多空（三种定义）", "",
              "| 定义 | 累计 |", "| :-- | --: |"] + \
        [f"| {k} | {v:.2%} |" for k, v in ls.items()] + [""]
    # 零依赖版 Spearman（pandas 那个要 scipy，requirements 里没有 ✗）
    mono = MT.spearman_rank(
        layer_tab["年化"].tolist(), list(range(len(layer_tab))))
    mono_txt = "nan" if mono is None else f"{mono:.3f}"
    parts += ["## 4. 结论（机械生成，不掺主观）", "",
              f"- 层号 vs 年化的 Spearman = **{mono_txt}**（越接近 1 越单调；< 0 说明高分位反而差）",
              f"- 最高层：层 {layer_tab['年化'].idxmax()}（年化 {layer_tab['年化'].max():.2%}）",
              f"- 最低层：层 {layer_tab['年化'].idxmin()}（年化 {layer_tab['年化'].min():.2%}）", ""]
    if note:
        parts += ["## 5. 备注", "", note, ""]
    parts += [f"![分层净值]({png.name})", ""]

    md.write_text("\n".join(parts), encoding="utf-8")
    return {"md": md, "png": png, "layer_tab": layer_tab, "ls": ls, "ic": ic, "ic_summary": ic_sum}


def main():
    from backtest.panel import load_panels
    from backtest.assemble import rebalance_dates
    from backtest.weights import weights_for_factor
    from backtest.engine import run_all_layers, nav_frame

    panels = load_panels()
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index) if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    res = run_all_layers(W, panels, rdates)
    out = build_report("rev_20d", nav_frame(res), panels, rdates, res)
    print("报告:", out["md"])
    print("图  :", out["png"])
    print(out["layer_tab"].round(4).to_string())


if __name__ == "__main__":
    main()
