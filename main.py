# -*- coding: utf-8 -*-
"""
程序入口 = 总调度
====================
职责：把 8 格串成一条链（panel → factor → weights → engine → metrics → report），
      逐因子跑全链并把报告写进 reports/。
      运行前先打印配置中心自检（"这次用的是哪份配置"）。

修正记录:
  2026-09-07  原 main.py 引用 config.loader（源文件不存在，仅残留旧 pyc，必报错）
              → 改走 config.config_manager（热加载单例），保持全仓库配置同源。
  2026-10-13  从"配置中心占位自检"升级为【总调度 v0】；原 main() 改名 config_check()，
              由新 main() 在开头调用 —— 此前两个函数同名，**后定义者静默覆盖前者**，
              结果跑的是旧自检（症状：能跑、有输出，但根本没跑调度）。
"""

from config.config_manager import config
import hashlib

import numpy as np
import pandas as pd
from pathlib import Path
from backtest.panel import load_panels
from backtest.assemble import build_features, rebalance_dates
from backtest.weights import build_target_weights, weights_for_factor
from backtest.engine import run_all_layers, nav_frame
from backtest.factors import FACTORS
from backtest import report as RP
from backtest.neutralize import neutralize
from loguru import logger
from config.research_spec import load_spec, save_snapshot


OUT = Path(__file__).resolve().parent/"reports"


LOG_PATH = Path(__file__).resolve().parent / "logs" / "run.log"


def init_log() -> None:
    """把每次运行的关键数字追加进 logs/run.log —— 见 教学文档/日志教学参考.md。

    固定格式：时间 | 因子 | 层0换手 | 末层换手 | 多空 | IC | n（便于两行并排比对）。
    末层的层号随 n_layers 变（5 层→层4、10 层→层9），所以标签写"层号"而不是写死"层4"。
    enqueue=True 是异步写（不拖慢大计算）；个别受限环境不能建命名管道时退回同步写 ——
    **日志绝不能让主流程失败**。
    """
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {message}"                 # 干净一行一条；异常时 loguru 自动追加 traceback
    # encoding="utf-8-sig"：带 BOM ⇒ Windows PowerShell 5.1 的 Get-Content / Select-String
    # 也能自动认出 UTF-8（否则默认按 ANSI 读 → 中文全乱码，2026-10-14 实测）
    try:
        logger.add(LOG_PATH, format=fmt, enqueue=True, encoding="utf-8-sig",
                   rotation="5 MB", retention="60 days")
    except Exception:
        logger.add(LOG_PATH, format=fmt, encoding="utf-8-sig",
                   rotation="5 MB", retention="60 days")


def main():
    # 运行前先证明"这次用的是哪份配置"
    config_check()
    init_log()                                                    # 关键数字落盘（日志教学参考 §5）
    panels = load_panels()  # 缓存提到循环外
    spec = load_spec(until=str(panels["hfq_close"].index[-1].date()))
    save_snapshot(spec)
    logger.info(f"RUN {spec.banner()}")

    # —— 运行身份（2026-10-14）：这一行让以后的任何一行数字都能回答"哪份代码 + 哪份数据"——
    _fp = hashlib.md5(panels["hfq_close"].to_numpy().tobytes()).hexdigest()[:8]
    logger.info(f"RUN 区间:{spec.since}+ 因子:{len(FACTORS)} panel:{_fp} "
                f"shape:{panels['hfq_close'].shape}")
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index, rule=spec.freq) if d.year >= spec.since]

    logger.info(f"RUN SINCE:{spec.since} 首个调仓日:{rdates[0].date()} 调仓:{len(rdates)}期 "
                f"因子:{len(FACTORS)} panel:{_fp} shape:{panels['hfq_close'].shape}")

    for i, name in enumerate(FACTORS, 1):
        print(f"[{i}/{len(FACTORS)}] {name} …", end=" ", flush=True)
        missing = [t for t in FACTORS[name].get(
            "requires", []) if t not in panels]
        if missing:                                               # 软失败：记下来、继续
            print(f"skipped  ⚠ 面板缺 {missing}")
            continue
        _spec = FACTORS[name]
        _fval = _spec["fn"](build_features(panels, [name]), {}) * _spec["sign"]

        # 股票池口径：剔除 ST（T-6.3）—— 因子值的【前置操作】，必须在一切消费之前。
        #   置 NaN 而不是改 can_trade：ST 股现实中能买卖，把它标成不可交易会让
        #   已持仓变 ST 的票「卖不掉」（假），还会污染 can_trade 这个市场事实字段。
        #   置 NaN 之后下游四处自动一致：中性化样本 / 权重 / rank_ic / 报告。
        if not spec.include_st:
            # 逐格写（PITFALLS #24：写入权不许外包给库）。用 .where() 会碰那个偶发出错的出口。
            _fv = np.array(_fval.to_numpy(dtype="float64"), copy=True)   # 一次快照
            _st = panels["is_st"].to_numpy()
            _fv[_st] = np.nan                                            # 整片写，不逐格循环
            _fval = pd.DataFrame(_fv, index=_fval.index, columns=_fval.columns)

        # 中性化（T-6.2）必须在这里 —— 在 build_target_weights 和 build_report【之前】：
        #   _fval 是唯一的因子值来源，它同时流向 ①权重 ②报告（报告内部调 rank_ic）。
        #   在这一行改，两者一起吃；
        #   若插在权重算完【之后】，权重仍是 raw 的、只有报告被中性化
        #   ⇒ 报告说「中性化后 IC 更高」但净值/分层根本没变，自相矛盾 ✗
        if spec.neutralize:
            _fval = neutralize(_fval, panels, rdates,
                               exposures=spec.neutralize_exposures,
                               mv_floor=spec.neutralize_mv_floor,
                               weight=spec.neutralize_weight,
                               drop_first=spec.neutralize_drop_first)

        W, _ = build_target_weights(
            _fval, rdates, panels["can_trade"], n_layers=spec.n_layers)

        res = run_all_layers(W, panels, rdates,
                             fee_buy=spec.fee_buy, fee_sell=spec.fee_sell,
                             slippage=spec.slippage, init_cash=spec.init_cash)

        # —— 哨兵（2026-10-13 建，2026-11-03 改）：换手塌陷才拒绝出报告 ——
        # 第三代判据：【年化换手】= 累计换手 ÷ 年数。为什么换了两次：
        #   第一代「累计 < 20」：随期数线性变，since=2020（81 期）时层0 只有 15.83 误报
        #   第二代「每期平均 < 15%」：随【频率】剧烈变，而且【不单调】——
        #       实测 amihud：ME 19.5% / QE 27.2% / W-FRI 8.0%
        #       （季度调仓让排名漂得更久 ⇒ 每期换手反而比月频高；周频最低）
        #       ⇒ 任何「每期 × 频率系数」的公式都套不上 ✗
        #   第三代「年化换手」：跨频率可比。实测 12 组（2 因子 × 3 频率 × 层0/层4）
        #       法定范围 1.090 ~ 8.958；#23 的损坏样本（累计 5 / 10.8 年）是 0.462
        #       ⇒ 取 0.7，两边各留 1.5 倍余量。
        _MIN_TV_PER_YEAR = 0.7
        _n_period = max(len(res[0]["turnover"]), 1)
        _years = _n_period / max(spec.periods_per_year, 1)
        _ann_tv = {L: float(res[L]["turnover"].sum()) / _years
                   for L in range(len(res))}
        # 窗口不足一年时年化换手没有统计意义（一次调仓就能算出 0.3），跳过不判
        low = [L for L, v in _ann_tv.items()
               if v < _MIN_TV_PER_YEAR] if _years >= 1.0 else []
        if low:
            try:
                # ── 2026-11-03：失败时把【上游指纹】全部 dump 出来 ──
                # 目的：下一次偶发时，能直接对比「失败这次」和「成功那次」的 md5，
                # 定位到底是 can_trade / fv / W / 引擎 哪一层先变的。
                # 只加日志，不改任何计算。
                def _m(x):
                    return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()[:10]
                _ct = panels["can_trade"].to_numpy()
                logger.warning(
                    f"DIAG panel  can_trade md5 {_m(_ct)} true_ratio {_ct.mean():.6f} "
                    f"hfq_close md5 {_m(panels['hfq_close'].to_numpy())}")
                _fvv = _fval.to_numpy()
                logger.warning(
                    f"DIAG fv     md5 {_m(_fvv)} shape {_fvv.shape} "
                    f"nonnan {int(np.isfinite(_fvv).sum())} "
                    f"min {np.nanmin(_fvv):.6g} max {np.nanmax(_fvv):.6g}")
                for _L in range(len(W)):
                    _w = W[_L].to_numpy()
                    logger.warning(
                        f"DIAG W{_L}     md5 {_m(_w)} nnz {int((_w != 0).sum())} "
                        f"rowsum_min {float(_w.sum(axis=1).min()):.6f}")
                _W0 = W[0].to_numpy(dtype="float64")
                _rs = _W0.sum(axis=1)
                logger.warning(
                    f"DIAG W[0] rowsum {_rs.min():.4f}/{_rs.max():.4f} nnz/row {(_W0 != 0).sum(axis=1).mean():.0f}")
                for _L in range(len(res)):
                    _tv = res[_L]["turnover"].to_numpy()
                    _tr = res[_L]["trades"].to_numpy()
                    _sh = res[_L]["shares"].to_numpy()
                    logger.warning(f"DIAG L{_L}: tv_sum {_tv.sum():.4f} tv_nz {int((_tv > 0).sum())}/{len(_tv)} "
                                   f"trades_nnz {int((_tr != 0).sum())} shares_last_nnz {int((_sh[-1] != 0).sum())} "
                                   f"nav_end {float(res[_L]['nav'].iloc[-1]):.6f}")
            except Exception as _e:
                logger.warning(f"DIAG failed: {_e}")
            _detail = "，".join(f"层{L}={_ann_tv[L]:.2f}/年" for L in low)
            raise AssertionError(
                f"年化换手异常低：{_detail}（阈值 {_MIN_TV_PER_YEAR:.1f}/年，"
                f"共 {_n_period} 期 ≈ {_years:.2f} 年，freq={spec.freq}）"
                f"-- result untrusted (possible nondeterministic data corruption, "
                f"PITFALLS #23); refusing to report")
        _nav_full = nav_frame(res)
        _nav = _nav_full.loc[rdates[0]:]  # ★ 唯一的新增：首个成交日之前的空仓期不进指标
        logger.info(f"nav 切片 {_nav_full.index[0].date()}~{_nav_full.index[-1].date()} "
                    f"=> 指标起算日 {_nav.index[0].date()} "
                    f"行数 {len(_nav_full)}→{len(_nav)} (drops {len(_nav_full) - len(_nav)})")

        out = RP.build_report(name, _nav, panels, rdates,
                              res, outdir=OUT, factor_val=_fval, split=spec.split, spec=spec)

        tv = [float(res[k]["turnover"].sum()) for k in range(len(res))]
        # 层号跟着 n_layers 走：原来写死"层4"，10 层时它其实是层9，日志会撒谎
        # （同类前车之鉴：报告 banner 曾把"月频"写死，改周频后印出"月频 553 期"）
        _last = len(res) - 1
        logger.info(f"{name} | 区间 {_nav.index[0].date()} | 层0换手 {tv[0]:.4f} "
                    f"| 层{_last}换手 {tv[-1]:.4f} | 多空净 {out['ls']['多空净年化']:.4f} "
                    f"| IC {out['ic_summary']['ic_mean']:.4f} "
                    f"| ICvalid {out['ic_summary']['n']} ICbad {out['ic_summary']['n_bad']}")

    logger.complete()          # 异步写：确保退出前把队列里的行落盘


def config_check() -> None:
    app_name: str = str(config.get("app_name", "Quant-1.0"))
    start: str = str(config.get("stock.fetch.start_date", "2015-01-01"))
    workers: int = int(config.get("stock.fetch.workers", 8))

    print(f"应用: {app_name}")
    print(f"品种: A股(含退市) | 日线起始 {start} | 下载并发 {workers} 进程")
    print("配置中心自检 OK — 交易主循环待 execution 模块接入")


if __name__ == "__main__":
    # —— 异常留痕（2026-10-14）：出事的 traceback 也要进日志 ——
    # 用 reraise=True：**日志记下来，退出码照样非 0**（不许把失败伪装成成功）
    try:
        with logger.catch(reraise=True, message="本次运行异常退出（traceback 见下）"):
            main()
    finally:
        logger.complete()          # 异步写：确保队列里的行（含上面那几行 traceback）落盘
