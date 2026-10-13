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
from pathlib import Path
from backtest.panel import load_panels
from backtest.assemble import rebalance_dates
from backtest.weights import weights_for_factor
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
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {message}"
    try:
        logger.add(LOG_PATH, format=fmt, enqueue=True,
                   rotation="5 MB", retention="60 days")
    except Exception:
        logger.add(LOG_PATH, format=fmt, rotation="5 MB", retention="60 days")


def main():
    # 运行前先证明"这次用的是哪份配置"
    config_check()
    init_log()                                                    # 关键数字落盘（日志教学参考 §5）
    panels = load_panels()                                        # 缓存提到循环外
    rdates = [d for d in rebalance_dates(
        panels["hfq_close"].index) if d.year >= SINCE]
    for i, name in enumerate(FACTORS, 1):
        print(f"[{i}/{len(FACTORS)}] {name} …", end=" ", flush=True)
        missing = [t for t in FACTORS[name].get(
            "requires", []) if t not in panels]
        if missing:                                               # 软失败：记下来、继续
            print(f"skipped  ⚠ 面板缺 {missing}")
            continue
        W, _ = weights_for_factor(name, panels, rdates)
        res = run_all_layers(W, panels, rdates)
        # —— 哨兵（2026-10-13）：结果落在合理区间才放行 ——
        # 依据：6 个因子 × 5 层的实测累计换手都在 44~103；若某层 < 20，多半是"买不进任何票"
        low = [L for L in range(len(res)) if float(
            res[L]["turnover"].sum()) < 20.0]
        if low:
            print(f"    ⚠ 结果可疑：层 {low} 累计换手 < 20 —— 本次运行的数字不可信，请重跑并比对")
        out = RP.build_report(name, nav_frame(
            res), panels, rdates, res, outdir=OUT)
        print(f"ok  {out['md'].name}")

        tv = [float(res[k]["turnover"].sum()) for k in range(len(res))]
        logger.info(f"{name} | 层0换手 {tv[0]:.4f} | 层4换手 {tv[-1]:.4f} "
                    f"| 多空 {out['ls']['比值法']:.4f} | IC {out['ic_summary']['ic_mean']:.4f} "
                    f"| n {out['ic_summary']['n']}")
    logger.complete()          # 异步写：确保退出前把队列里的行落盘（日志教学参考 §8）


def config_check() -> None:
    app_name: str = str(config.get("app_name", "Quant-1.0"))
    start: str = str(config.get("stock.fetch.start_date", "2015-01-01"))
    workers: int = int(config.get("stock.fetch.workers", 8))

    print(f"应用: {app_name}")
    print(f"品种: A股(含退市) | 日线起始 {start} | 下载并发 {workers} 进程")
    print("配置中心自检 OK — 交易主循环待 execution 模块接入")


if __name__ == "__main__":
    main()
