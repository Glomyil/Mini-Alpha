"""ops/oos_compare.py —— T-3.8 样本外复现性检验（v2 判据）

判据（2026-11-13 定）：
  ① 方向      验证段 IC > 0（因子值已含冻结的 sign）
  ② IC-IR     验证 ≥ 0.5 × 训练
  ③ 多空 t    验证段 t_nw ≥ 1.5
  ④ p         双条件：p(多空t) < 0.05 且 p(形状重合ρ) < 0.05
  ⑤ IC 同号   两段 IC 均值同号
  已删除：分层单调（实测 6/6 因子二次项 c<0 为鼓形，与该判据要求反向）

不含年化/超额/卡玛 —— 收益表现在别的环节判，不在这里

为什么默认 10 层：5 层会把最差 10% 的信号稀释一半（实测 D0 −7.2% → 层0 −1.26%）

用法
    python -m ops.oos_compare                  # 用 yml 的 research.cut
    python -m ops.oos_compare --cut 2020-01-02 --factors amihud_20d
    python -m ops.oos_compare --verbose        # 想看库日志时加
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from backtest import metrics as MT
from backtest.assemble import rebalance_dates
from backtest.factors import FACTORS
from backtest.panel import load_panels
from backtest.pool import make_pool
from config.research_spec import load_spec
from main import build_factor_nav          # 同一份管线（单一来源）


def _period_rets(nav, rd):
    """逐期层收益矩阵 (n_periods, n_layers)。"""
    idx = [d for d in rd if d in nav.index]
    rows = []
    for i in range(len(idx) - 1):
        r = (nav.loc[idx[i + 1]] / nav.loc[idx[i]] - 1.0).to_numpy(dtype=float)
        if np.all(np.isfinite(r)):
            rows.append(r)
    return np.asarray(rows), idx[:-1]


def _stats(name, panels, spec, rd, pool):
    """跑一段，返回五条判据需要的量。"""
    r = build_factor_nav(name, panels, spec, rd, pool)
    nav, f = r["nav"], r["fval"]
    use = [d for d in rd if d in nav.index]
    ic = MT.rank_ic(f, panels, use)
    s = MT.ic_summary(ic, periods_per_year=spec.periods_per_year)
    R, _ = _period_rets(nav, use)
    pr = nav.loc[use]
    top, bot = pr[pr.columns[-1]], pr[pr.columns[0]]
    ls = (top / top.iloc[0]) / (bot / bot.iloc[0])
    ls_r = ls.pct_change().dropna().to_numpy()
    t_nw, p_t = MT.t_stat_nw(ls_r) if ls_r.size > 2 else (
        float("nan"), float("nan"))
    return {
        "n": len(use),
        "ic": float(s["ic_mean"]),
        "ic_ir": float(s["ic_ir"]),
        "t_nw": float(t_nw),
        "p_t": float(p_t),
        "R": R,
    }


def _rho_and_p(R1, R2, n_boot=2000, seed=7):
    """两段平均层收益向量的相关 + bootstrap 单侧 p（H0: ρ<=0）。

    为什么不用朴素 p：10 个层的相邻层高度相关，不是 10 个独立观测，
    朴素检验会把显著性吹大。重采样在【期】这一维上做。
    """
    a, b = R1.mean(axis=0), R2.mean(axis=0)
    if a.std() == 0.0 or b.std() == 0.0:
        # ρ 数学上未定义。必须抛错而不是返回 NaN：下游判据是 p_rho < 0.05，
        # NaN 比较恒为 False ⇒ 会静默把因子判成"未复现"，而真正的问题是数据退化。
        raise ValueError("层收益在某一段里方差为 0（ρ 未定义）—— 数据退化，不是因子失效")
    rho = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        x = R1[rng.integers(0, len(R1), len(R1))].mean(axis=0)
        y = R2[rng.integers(0, len(R2), len(R2))].mean(axis=0)
        boot[k] = np.corrcoef(x, y)[0, 1]
    p = float((boot <= 0).mean())
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return rho, p, float(lo), float(hi)


def _judge(name, tr, va, rho, p_rho, lo, hi):
    """六条检查 -> (是否复现, checks)。checks 元素 = (名, 过否, 通过时文案, 失败时文案)。"""
    sign = int(FACTORS[name]["sign"])
    ratio = (va["ic_ir"] / tr["ic_ir"]) if tr["ic_ir"] else float("nan")
    checks = [
        ("① 方向",    va["ic"] > 0,
         "验证 IC %+.4f（sign %+d）" % (va["ic"], sign),
         "验证 IC %+.4f ≤ 0" % va["ic"]),
        ("② IC-IR",   va["ic_ir"] >= 0.5 * tr["ic_ir"],
         "%.2f× 训练（%.3f vs %.3f）" % (ratio, va["ic_ir"], tr["ic_ir"]),
         "%.2f× 训练 = %.3f < 0.5 × %.3f" % (ratio, va["ic_ir"], tr["ic_ir"])),
        ("③ 多空 t",  va["t_nw"] >= 1.5,
         "%.2f" % va["t_nw"],
         "%.2f < 门槛 1.5" % va["t_nw"]),
        ("④a p(t)",   va["p_t"] < 0.05,
         "%.4f" % va["p_t"],
         "%.4f ≥ 门槛 0.05" % va["p_t"]),
        ("④b p(ρ)",   p_rho < 0.05,
         "%.4f（ρ %+.3f）" % (p_rho, rho),
         "%.4f ≥ 门槛 0.05（ρ %+.3f，CI [%+.2f, %+.2f]）" % (p_rho, rho, lo, hi)),
        ("⑤ IC 同号", bool(np.sign(va["ic"]) == np.sign(tr["ic"])),
         "训练 %+.4f / 验证 %+.4f" % (tr["ic"], va["ic"]),
         "两段符号相反：训练 %+.4f / 验证 %+.4f" % (tr["ic"], va["ic"])),
    ]
    return all(c[1] for c in checks), checks


def _row(name, tr, va, checks):
    """一行的定宽文本：因子 / 六列 ✓✗ / 判决。"""
    def mark(c): return "✓" if c[1] else "✗"
    ratio = (va["ic_ir"] / tr["ic_ir"]) if tr["ic_ir"] else float("nan")
    return ("  %-12s  %s      %6s    %5s   %6.4f   %6.4f    %s     %s"
            % (name, mark(checks[0]),
               "%.2f×" % ratio if tr["ic_ir"] else "n/a",
               "%.2f" % va["t_nw"], va["p_t"], va["p_rho"] if "p_rho" in va else float(
                   "nan"),
               mark(checks[5]),
               "复现 " if all(c[1] for c in checks) else "未复现 "))


def main() -> None:
    ap = argparse.ArgumentParser(description="T-3.8 样本外复现性检验")
    ap.add_argument("--cut", default="", help="切点（空 = 用 yml 的 research.cut）")
    ap.add_argument("--factors", default="", help="逗号分隔；空 = 全部")
    ap.add_argument("--layers", type=int, default=10, help="分层数（两段一致）")
    ap.add_argument("--verbose", action="store_true",
                    help="保留库日志（默认静音，否则表格被打散）")
    args = ap.parse_args()

    if not args.verbose:
        logger.remove()

    panels = load_panels()
    spec = load_spec(n_layers=args.layers)
    cut_str = args.cut or spec.cut
    if not cut_str:
        raise SystemExit("没有切点：命令行 --cut 或 yml 的 research.cut 至少给一个")
    pool = make_pool(panels, spec)
    all_rd = [d for d in rebalance_dates(panels["hfq_close"].index, rule=spec.freq)
              if d.year >= spec.since]
    cut = pd.Timestamp(cut_str)
    train_rd = [d for d in all_rd if d <= cut]
    valid_rd = [d for d in all_rd if d >= cut]
    if not train_rd:
        raise SystemExit("切点 %s 之前没有调仓日" % cut.date())

    spec_tr = load_spec(n_layers=args.layers, until=str(train_rd[-1].date()))
    spec_va = load_spec(n_layers=args.layers, since=cut.year)
    a, b = spec_tr.fingerprint().split("|"), spec_va.fingerprint().split("|")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    keys = [x.split("=")[0] for x, _ in diff]
    if not (len(a) == len(b) and set(keys) <= {"since", "until"}):
        raise SystemExit("两段 spec 只允许差 since/until，实际差：%r" % (diff,))

    names = [x.strip() for x in args.factors.split(",")
             if x.strip()] or list(FACTORS)
    rows, fails = [], []
    for i, name in enumerate(names, 1):
        print("  计算中 %d/%d  %-12s" %
              (i, len(names), name), end="\r", flush=True)
        tr = _stats(name, panels, spec_tr, train_rd, pool)
        va = _stats(name, panels, spec_va, valid_rd, pool)
        rho, p_rho, lo, hi = _rho_and_p(tr["R"], va["R"])
        ok, checks = _judge(name, tr, va, rho, p_rho, lo, hi)
        va["p_rho"] = p_rho
        rows.append(_row(name, tr, va, checks))
        if not ok:
            fails.append((name, [c for c in checks if not c[1]]))
    print(" " * 40, end="\r")

    print()
    print("  切点 %s → 训练末 %s ｜ %d 层 ｜ 训练 %d 期 / 验证 %d 期"
          % (cut.date(), train_rd[-1].date(), args.layers, len(train_rd), len(valid_rd)))
    print("  两段 spec 校验 ✓（只差 %s）" % ", ".join(keys))
    print()
    print("  因子          ①方向  ②IC-IR   ③多空t  ④a p    ④b p    ⑤同号  判决")
    print("  " + "-" * 76)
    for r in rows:
        print(r)

    if fails:
        print()
        print("  未复现明细")
        for name, bad in fails:
            print("    %s" % name)
            for label, _, _, why in bad:
                print("      %-9s %s" % (label, why))
    print()
    print("  记账写进 docs/dev/oos/ledger.md（只记 ✓/✗，不记收益）")


if __name__ == "__main__":
    main()
