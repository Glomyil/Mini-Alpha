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
from pathlib import Path
from backtest.panel import load_panels
from backtest.assemble import build_features, rebalance_dates
from backtest.weights import build_target_weights, weights_for_factor
from backtest.engine import run_all_layers, nav_frame
from backtest.factors import FACTORS
from backtest import report as RP
from loguru import logger

OUT = Path(__file__).resolve().parent/"reports"

SINCE = 2016

LOG_PATH = Path(__file__).resolve().parent / "logs" / "run.log"


def init_log() -> None:
    """把每次运行的关键数字追加进 logs/run.log —— 见 教学文档/日志教学参考.md。

    固定格式：时间 | 因子 | 层0换手 | 层4换手 | 多空 | IC | n（便于两行并排比对）。
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
    panels = load_panels()                                        # 缓存提到循环外
    # —— 运行身份（2026-10-14）：这一行让以后的任何一行数字都能回答"哪份代码 + 哪份数据"——
    _fp = hashlib.md5(panels["hfq_close"].to_numpy().tobytes()).hexdigest()[:8]
    logger.info(f"RUN 区间:{SINCE}+ 因子:{len(FACTORS)} panel:{_fp} "
                f"shape:{panels['hfq_close'].shape}")
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index) if d.year >= SINCE]

    logger.info(f"RUN SINCE:{SINCE} 首个调仓日:{rdates[0].date()} 调仓:{len(rdates)}期 "
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
        W, _ = build_target_weights(_fval, rdates, panels["can_trade"])
        res = run_all_layers(W, panels, rdates)
        # —— 哨兵（2026-10-13）：结果落在合理区间才放行 ——
        # 依据：6 个因子 × 5 层的实测累计换手都在 44~103；若某层 < 20，多半是"买不进任何票"
        low = [L for L in range(len(res)) if float(
            res[L]["turnover"].sum()) < 20.0]
        if low:
            try:
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
            raise AssertionError(
                f"layer {low} cumulative turnover < 20 -- result untrusted "
                f"(possible nondeterministic data corruption, PITFALLS #23); refusing to report")
        _nav_full = nav_frame(res)
        _nav = _nav_full.loc[rdates[0]:]  # ★ 唯一的新增：首个成交日之前的空仓期不进指标
        logger.info(f"nav 切片 {_nav_full.index[0].date()}~{_nav_full.index[-1].date()} "
                    f"=> 指标起算日 {_nav.index[0].date()} "
                    f"行数 {len(_nav_full)}→{len(_nav)} (drops {len(_nav_full) - len(_nav)})")

        out = RP.build_report(name, _nav, panels, rdates,
                              res, outdir=OUT, factor_val=_fval)

        tv = [float(res[k]["turnover"].sum()) for k in range(len(res))]
        logger.info(f"{name} | 区间 {_nav.index[0].date()} | 层0换手 {tv[0]:.4f} "
                    f"| 层4换手 {tv[-1]:.4f} | 多空 {out['ls']['比值法']:.4f} "
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
