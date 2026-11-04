from __future__ import annotations
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from backtest import metrics as MT
from config.research_spec import freq_label, universe_label
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
        # 「月频」曾经是字面量 ⇒ 改 freq 后期数变了、标签没变（月频 553 期 = 46 年）。
        # 现在跟着 meta['freq'] 走；未知代码原样印出，不猜（PITFALLS #19）。
        f"> - 因子：`{meta['factor']}` ｜ 区间：{meta['start']} ~ {meta['end']} ｜ "
        # 硬索引，不用 .get(默认值)：本函数其余字段都硬索引，
        # 缺键就该 KeyError（docstring：「缺一项就算没做这一格」）。
        f"调仓：{freq_label(meta['freq'])}（{meta['n_rdate']} 期）",
        f"> - 股票池：{universe_label(meta['universe'])}",
        f"> - 净值口径：**{meta['val_basis']}**（收盘 / 开盘两套，另一套作诊断）｜ 成交：**T+1 开盘**（缺价即不成交）",
        f"> - 年化常数：**{meta['trading_days']}** 交易日/年（A 股；252 是美股）｜ IC-IR 用 {meta['periods_per_year']} 期/年",
        f"> - 成本：佣金 {meta['fee_buy']}/{meta['fee_sell']} ｜ 滑点 {meta['slippage']}（单边）｜ 无整手",
        f"> - 多空：**标准复利 ∏(1+r₄−r₀)**（主）· 层差 pp · 扣融券净额 —— 多空是**纯度诊断**，不是可执行策略",
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
        ax.plot(nav_df.index, nav_df[L].to_numpy(),
                lw=1.2, alpha=0.85, label=f"层 {L}")
    if bench is not None:
        ax.plot(bench.index, bench.to_numpy(), lw=2.4, color="#2c7fb8",
                label="基准(全A等权)")
    ax.axhline(1.0, color="#9aa5b1", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("净值（对数轴，起点 = 1.0）")

    # 副轴画的是【层4 净值 ÷ 层0 净值】—— 它是"相对强弱"，不是多空收益。
    # 原图例写"多空(比值法)"会让人以为这是多空组合的钱，改回它本来的名字。
    ratio = nav_df[nav_df.columns[-1]] / nav_df[nav_df.columns[0]]
    ax2 = ax.twinx()
    ax2.plot(nav_df.index, ratio.to_numpy(), lw=1.2, ls="--", color="#c0392b",
             alpha=0.55, label="层4÷层0 净值比（非多空收益）")
    ax2.set_ylabel("层4÷层0 净值比", fontsize=8, color="#c0392b")
    ax2.tick_params(labelsize=7, colors="#c0392b")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=4, fontsize=8,
              frameon=False, loc="upper left")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report(factor, nav_df, panels, rdates, res=None, *, rf=0.0, outdir: Path = OUT,
                 val_basis: str = "close", note: str = "", factor_val=None,
                 split: str = "", spec=None) -> dict:
    # ★ trading_days 是唯一源头：报告表头、年化、波动、夏普、层差全用它。
    #   不传的话 metrics 会退回自己的 244 常量 ⇒ 表头印 252 但数字按 244 算（PITFALLS #19）。
    _td = spec.trading_days if spec else MT.TRADING_DAYS
    layer_tab = MT.layer_table(nav_df, res, rf=rf, trading_days=_td)

    # ── T-3.1 基准线：全 A 等权（每日再平衡、未扣费，基准偏高 => 超额偏保守）──
    _bench = None
    if panels is not None and "ret_1d_hfq" in panels and "can_trade" in panels:
        _bench = MT.equal_weight_nav(panels).reindex(nav_df.index).ffill()
        # 归一化到 1.0：它从 2015 年累乘起，直接叠到 2016 起的 nav 上会从 1.6 开始。
        # 年化不受影响（ann_return 是相对算的），但图上起点不一致等于骗人 ✗
        _bench = _bench / float(_bench.iloc[0])
        _bann = MT.ann_return(_bench, _td)
        layer_tab["基准年化"] = _bann
        layer_tab["超额"] = layer_tab["年化"] - _bann
    else:
        layer_tab["基准年化"] = "缺基准数据"
        layer_tab["超额"] = "缺基准数据"

    # freq 是唯一源头：年化粒度用 spec 的 periods_per_year，不写死 12
    _bc = spec.borrow_cost if spec else 0.08
    _ppy = spec.periods_per_year if spec else MT.REBALANCE_PERIODS_PER_YEAR
    ls = MT.ls_summary(nav_df, rdates, borrow_cost=_bc, periods_per_year=_ppy,
                       trading_days=_td)

    ic = ic_sum = None
    if panels is not None and factor is not None:
        # not provided -> compute it once (legacy path)
        if factor_val is None:
            from backtest.assemble import build_features
            from backtest.factors import FACTORS
            # ⚠️ 变量名不能叫 spec —— 那是本函数的形参（RunSpec 或 None），
            #    覆盖之后下面 meta 里的 spec.trading_days 就会 AttributeError。
            #    只在 factor_val=None 的旧路径（test_scheduler / metrics.main）发作，
            #    main.py 永远传 factor_val ⇒ 之前完全隐身。
            _fspec = FACTORS[factor]
            factor_val = _fspec["fn"](build_features(
                panels, [factor]), {}) * _fspec["sign"]
        ic = MT.rank_ic(factor_val, panels, rdates)
        # IC-IR 的年化粒度必须跟 freq 走（和 banner 印的那个数保持一致）
        ic_sum = MT.ic_summary(ic, periods_per_year=_ppy)
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
            "n_rdate": len(rdates),
            "trading_days": spec.trading_days if spec else MT.TRADING_DAYS,
            "periods_per_year": spec.periods_per_year if spec else MT.REBALANCE_PERIODS_PER_YEAR,
            # freq 是报告里「调仓：月频/季频/周频」的唯一来源。不传 spec 时给 "?"
            # 而不是 "ME" —— 猜一个就等于把月频假设藏起来（PITFALLS #19）。
            "freq": spec.freq if spec else "?",
            # 股票池：和 freq 一样，不传 spec 就给 "?"，不猜 "all"（PITFALLS #19）
            "universe": spec.universe if spec else "?",
            "val_basis": val_basis,
            "fee_buy": spec.fee_buy if spec else 0.00025,
            "fee_sell": spec.fee_sell if spec else 0.00075,
            "slippage": spec.slippage if spec else 0.0005,
            "stamp": stamp}

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

        # 每期损失 × 每年期数 = 年化；期数由 freq 决定，不能写死 12
        pp = MT._blocked_pp(_top, panels, periods_per_year=_ppy)
        parts += [
            "## 0. 成交失败（P2 核心结论）", "",
            f"- 目标买入里，**{n_lu} 笔撞上涨停**没买成（共 {n_ex} 期）",
            f"- 目标卖出里，**{n_ld} 笔撞上跌停**没卖成",
            f"- 另有 **{n_buy - n_lu} 笔**因停牌 / 不在市 / 无价没买成",
            f"- 这些没成交的单子，一年大约 **{pp:.2f} pp** 收不到",
            "",
        ]

    # 「每期换手」必须按百分比渲染（0.5000 → 50.00%）：它进这张表就是为了
    # 让人看出「每期换掉一半仓位」，印成 0.5000 等于没达到目的。
    parts += ["## 1. 分层指标", "",
              md_table(layer_tab, pct_cols=("年化", "波动", "最大回撤",
                                            "基准年化", "超额", "每期换手")), "",
              "*「佣金合计」只含佣金与印花税。**滑点不进这一列** —— 它在成交价里"
              "（买入多付、卖出少收），已经体现在净值和年化上。*", ""]

    # ── T-3.3 §2 分段对比：split 为空则整节跳过（空转也留痕，PITFALLS 1096）──
    seg = None
    if split:
        seg = segment_table(nav_df, split, bench=_bench, trading_days=_td)
        if len(seg) == 0:
            logger.warning(f"{factor} 分段表为空：split={split!r} 未产生任何段")
        else:
            parts += ["## 2. 分段对比", "",
                      f"> 分界点：`{split}`（来自 application.yml 的 research.split）", "",
                      md_table(seg.set_index("段"), index_name="段",
                               pct_cols=("层0年化", "层0超额",
                                         "层4年化", "层4超额")), ""]
    else:
        logger.info(f"{factor} 分段跳过：split 为空（不分段模式）")

    # ── T-3.3 §3 年度 IC：ic 已在作用域里（rank_ic 返回的就是逐期 Series）──
    yic = None
    if ic is not None and len(ic) > 0:
        yic = yearly_ic_table(ic)
        parts += ["## 3. 年度 IC（标签 = 下一个调仓日的区间收益）", "",
                  md_table(yic, index_name="年",
                           pct_cols=("IC均值", "IC胜率")), ""]
    if ic_sum is not None:
        # IC>0 胜率：只有均值会飘（几期大值就能撑起来），胜率才说明方向稳不稳。
        # 有效期数为 0 时 ic_pos_rate 是 NaN ⇒ 印 nan，不许崩（报告不该因 NaN 挂掉）。
        _pos = ic_sum["ic_pos_rate"]
        if _pos != _pos:
            _pos_txt = "- **IC > 0 胜率 nan**（没有有效 IC 期，无法计算）"
        else:
            _pos_txt = (f"- **IC > 0 胜率 {_pos:.1%}**"
                        f"（{ic_sum['n_pos']} / {ic_sum['n']} 期为正）")
        logger.info(f"ic 有效:{ic_sum['n']} 无效:{ic_sum.get('n_bad', 0)} "
                    f"mean:{ic_sum['ic_mean']:.4f} ir:{ic_sum['ic_ir']:.4f}")
        parts += ["## 4. IC（标签 = 下一个调仓日的区间收益）", "",
                  f"- IC 均值 **{ic_sum['ic_mean']:.4f}** ｜ 标准差 {ic_sum['ic_std']:.4f} "
                  f"｜ **IC-IR {ic_sum['ic_ir']:.4f}** ｜ 期数 {ic_sum['n']}",
                  _pos_txt,
                  ""]
    # ── §5 多空诊断（2026-11-03 重写：删比值法/日差法/月差法，改标准复利 + t/p）──
    _t, _p = ls["多空t"], ls["多空p"]
    _t_txt = "nan" if _t != _t else f"{_t:+.2f}"          # x != x 是 nan 的判据
    _p_txt = "nan" if _p != _p else f"{_p:.3g}"
    _sh, _mdd = ls["多空夏普"], ls["多空最大回撤"]
    _sh_txt = "nan" if _sh != _sh else f"{_sh:+.2f}"
    _mdd_txt = "nan" if _mdd != _mdd else f"{_mdd:.2%}"
    parts += ["## 5. 多空诊断（不可直接交易）", "",
              "> 多空不是交易策略 —— A 股个股做空基本不可行（券源少、费率高、很多票没券）。",
              "> 它是「因子纯度的体温计」：剥离市场 beta 之后还剩多少 alpha。",
              "> **你能实际赚的是「多头超额」，见 §1 的「超额」列和 §6 第一条。**", "",
              "| 口径 | 年化 | 累计 | 假设 |",
              "| :-- | --: | --: | :-- |",
              # ⚠️ 层差的单位是【百分点】不是百分比：0.1489 要印成 "+14.89 pp"，
              #   不能印成 "+14.89%"（那会被读成"涨了 14.89%"）。这正是本票在治的口径误读。
              f"| 层差（pp） | {ls['层差(pp)'] * 100:+.2f} pp | — | 两个独立多头组合，不做空 |",
              f"| 多空（零成本） | {ls['多空年化']:+.2%} | {ls['多空累计']:+.2%} "
              f"| 每期再平衡、零成本做空、券源无限 |",
              f"| **多空净** | **{ls['多空净年化']:+.2%}** | — "
              f"| 扣 {_bc:.0%}/年融券费 |",
              "",
              # 多空夏普 / 最大回撤（2026-11-04）：只有均值+t 不够，得看路径。
              #   ⚠️ 这两个都基于【每期】差值序列，年化用 periods_per_year，不是 trading_days。
              f"> 多空夏普 **{_sh_txt}** ｜ 最大回撤 **{_mdd_txt}**"
              f"（零成本口径，按 {ls['期数']} 期）",
              f"> 配对 t = **{_t_txt}** ｜ p = **{_p_txt}**"
              f"（n = {ls['期数']} 期，df = {ls['期数'] - 1}，p 用正态近似）",
              "> 「层差」是两个年化率相减（pp），**不存在累计口径**，所以那格写「—」。", ""]
    # 零依赖版 Spearman（pandas 那个要 scipy，requirements 里没有 ✗）
    mono = MT.spearman_rank(
        layer_tab["年化"].tolist(), list(range(len(layer_tab))))
    mono_txt = "nan" if mono is None else f"{mono:.3f}"
    # ★ 第一句放「你能实际赚的」—— 多空是诊断，超额才是钱
    _ex = layer_tab["超额"].iloc[-1] if "超额" in layer_tab else None
    _ex_txt = f"{_ex:+.2%}" if isinstance(_ex, float) and _ex == _ex else "缺基准数据"
    _hi = layer_tab["年化"].idxmax()
    parts += ["## 6. 结论（机械生成，不掺主观）", "",
              f"- **可交易：层 {_hi} 相对全 A 等权超额 {_ex_txt}**"
              f"（只做多是 A 股唯一能做的方向）",
              f"- 多空净年化 {ls['多空净年化']:+.2%}（扣 {_bc:.0%} 融券后）"
              f"—— 用于判断 alpha 纯不纯，不是可执行策略",
              f"- 层号 vs 年化的 Spearman = **{mono_txt}**（越接近 1 越单调；< 0 说明高分位反而差）",
              f"- 最高层：层 {_hi}（年化 {layer_tab['年化'].max():.2%}）",
              f"- 最低层：层 {layer_tab['年化'].idxmin()}（年化 {layer_tab['年化'].min():.2%}）", ""]
    if note:
        parts += ["## 7. 备注", "", note, ""]
    parts += [f"![分层净值]({png.name})", ""]

    md.write_text("\n".join(parts), encoding="utf-8")
    return {"md": md, "png": png, "layer_tab": layer_tab, "ls": ls,
            "ic": ic, "ic_summary": ic_sum,
            "segment": seg, "yearly_ic": yic}


def main():
    from backtest.panel import load_panels
    from backtest.assemble import rebalance_dates
    from backtest.weights import weights_for_factor
    from backtest.engine import run_all_layers, nav_frame

    panels = load_panels()
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index, rule="ME") if d.year >= 2016]
    W, layers = weights_for_factor("rev_20d", panels, rdates)
    res = run_all_layers(W, panels, rdates)
    out = build_report("rev_20d", nav_frame(res), panels, rdates, res)
    print("报告:", out["md"])
    print("图  :", out["png"])
    print(out["layer_tab"].round(4).to_string())


def segment_table(nav_df, splits, *, bench=None, layers=None,
                  trading_days: int = MT.TRADING_DAYS):
    splits = [s.strip() for s in str(splits).split(",") if s.strip()]
    idx = nav_df.index
    if not splits:
        return pd.DataFrame()          # 不分段 ⇒ 空表，调用方据此跳过整节

    # ★ layers 默认取【首层 + 末层】而不是写死 (0, 4)：
    #   n_layers 参数化之后，10 层时的「层4」是【中间层】，写死会把中间层当顶层用，
    #   于是同一份报告里 §2 的「顶层」和 §1/§5/§6 的「顶层」不是同一层。
    if layers is None:
        layers = (nav_df.columns[0], nav_df.columns[-1])

    # 边界对齐：split 必须落在 idx 上，落在两期之间取前一期
    cuts = []
    for s in splits:
        t = pd.Timestamp(s)
        if t < idx[0] or t > idx[-1]:
            raise ValueError(
                f"split={s} 不在回测区间 {idx[0].date()}~{idx[-1].date()} 内")
        pos = idx.searchsorted(t, side="right") - 1
        cuts.append(idx[pos])

    bounds = [idx[0]] + cuts + [idx[-1]]
    rows = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg = nav_df.loc[a:b]
        if len(seg) < 2:
            rows.append({"段": f"{i+1}", "起": a.date(), "止": b.date(),
                         "天数": len(seg), "备注": "数据不足"})
            continue
        row = {"段": f"{i+1}", "起": a.date(), "止": b.date(), "天数": len(seg)}
        for L in layers:
            s = seg[L]
            norm = s / s.iloc[0]        # ★ 例外③：段内归一化
            row[f"层{L}年化"] = MT.ann_return(norm, trading_days)   # ★ 独立算
            if bench is not None:
                bs = bench.loc[a:b]
                bn = bs / bs.iloc[0]    # ★ 基准也归一化
                row[f"层{L}超额"] = (MT.ann_return(norm, trading_days)
                                  - MT.ann_return(bn, trading_days))
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_ic_table(ic_series) -> pd.DataFrame:
    """按年分组算 IC 均值 / 期数 / 正比例。

    ic_series : pd.Series  逐期 IC（index 是调仓日）
    """
    df = ic_series.dropna()
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        "期数":   g.size(),
        "IC均值": g.mean(),
        "IC胜率": g.apply(lambda s: float((s > 0).mean())),
    })
    return out


if __name__ == "__main__":
    main()
