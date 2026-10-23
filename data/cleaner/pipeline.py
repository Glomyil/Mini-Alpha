"""pipeline.py —— 执行器：把 df 按清单顺序递给每个清洗环节，并收集报告

职责：读环节清单 → 逐只取 df(loader)→ 按顺序叫号 → 记录每步指标 → 汇总
不做：任何清洗逻辑；不排序（顺序 = 清单顺序）；不读 parquet(loader 的活)

跑法：
    python -m data.cleaner.pipeline      # 手动冒烟：前 3 只、2020 区间
扩展：
    写完一个环节，就在 PIPELINE 里加一行 —— 其它地方都不用动
"""
import sys

from loguru import logger

from config import constants as C
from data import loader
from data.cleaner import base
from data.cleaner import align, suspension, price, adjust, derived
# 环节清单：顺序 = 执行顺序；元素 = (环节名, 模块)
PIPELINE = [("align", align), ("suspension", suspension), ("price", price), ("adjust", adjust), ("derived", derived)
            # ("align", align),                   # 这里填充对应的模块名称，方便后续效用
            ]

# 守卫：base.PIPELINE_NAMES 是给"不能依赖本模块的代码"用的名字常量（如 backtest/panel.py）
#       两边漂了 ⇒ 缓存身份会误判 ⇒ 必须当场炸，不能静默
assert base.PIPELINE_NAMES == tuple(n for n, _ in PIPELINE), (
    f"PIPELINE_NAMES 与 PIPELINE 不一致：{base.PIPELINE_NAMES} vs {tuple(n for n, _ in PIPELINE)}")

_START, _END = "2020-01-01", "2020-12-31"


def run_one(df, ctx, steps=None):
    """让一只股票的 df 依次走过清单里的每个环节 → (df, 每步报告)"""
    # ctx 契约：环节会读这两个键，缺了当场报错（别等到环节里 KeyError）
    for key in ("asset_type", "code"):  # 用于标记对应数据的类型并检测对应的列
        if key not in ctx:
            # 若数据出现错误，或报错并反应对应缺失列
            raise KeyError(f"ctx 缺必填键 '{key}'（现有: {list(ctx)}）")

    reports = []
    for name, mod in (PIPELINE if steps is None else steps):
        before_cols = list(df.columns)          # 记账：进环节前的列（快照）
        rows_before = len(df)                   # 记账：进环节前的行数

        out = mod.run(df, {}, ctx)              # 叫号：把 df 递给环节
        if out is None:
            # run函数未返回处理结果自动报错
            raise TypeError(f"环节 '{name}' 返回了 None —— 忘了 return df？")
        df = out  # 处理后的dataframe代替out并产出

        rep = base.new_step_report(name)        # 报告卡（base 预置了契约字段）
        rep["metrics"] = {"rows_before": rows_before, "rows_after": len(df)}
        rep["columns_added"] = [c for c in df.columns if c not in before_cols]
        extra = base.check_new_columns(before_cols, list(df.columns))
        if extra:
            base.add_warning(rep, f"出现未登记列（不在白名单）: {extra}")
        reports.append(rep)
    return df, reports  # 终输出步骤报错，记录故障数据


def run_pipeline(codes, start, end, steps=None):  # 定义了一个可以遍历codes内部编号进行整体处理的函数
    """逐只股票跑完整条线"""
    for code, df in loader.iter_frames(codes, start, end):  # 由于loader的这个函数用yield控制进程所以会先一条数据跑完pipeline后再导入下一个dataframe
        ctx = {"universe": "stock", "asset_type": "stock",
               "code": code}  # 这里标注了股票对应的类型
        df, reports = run_one(df, ctx, steps)  # 现在开始处理一个股票对应的数据

        desc = " | ".join(f"{r['name']} {r['metrics']['rows_before']}"
                          # 用于展现数据清洗的进度，便于追溯
                          f"→{r['metrics']['rows_after']} +{len(r['columns_added'])}列"
                          for r in reports) or "（无环节）"
        print(f"  {code}: {desc}")  # 最终展现进度


# ---------------------------------------------------------------------------
# 总调度（2026-10-16）：数据更新那天，一条命令跑完全链
#     用法：python -m data.cleaner.pipeline all
#     口径：① 段级 —— 任一步失败就【停下】并说明"后续已跳过"（不许跳过继续 ✗）
#           ② 不做跳过判据（数据更新频率低；clean 每次都跑，正好堵住"忘了洗" ✓）
#           ③ fetch 不进这里（联网/慢/自有命令行）⇒ 更新源数据时先手动跑一次 fetch
# ---------------------------------------------------------------------------
_ALL_START, _ALL_END = "2015-01-01", "2026-12-31"


def _step_clean(codes, start, end, out_dir):
    """① 清洗落盘：直接复用今天新写的 clean_to_disk（本文件一行清洗逻辑都没有 ✓）"""
    from data.cleaner import clean_to_disk        # 延迟 import：避免与它互相 import
    meta = clean_to_disk.run(codes, start, end, out_dir)
    if meta["failures"]:                          # 只级失败：出声，但不中断 ✓
        logger.warning(f"clean 有 {len(meta['failures'])} 只失败，样例：{meta['failures'][:3]}")
    if codes and meta["stocks"] == 0:             # 一只都没成 ⇒ 段级问题（写不了盘/读不了源）✗
        raise RuntimeError(
            f"clean 一只都没写成功（{len(meta['failures'])} 只失败）—— 拒绝往下走，"
            f"样例：{meta['failures'][:3]}")

    # 账本只由「全库运行」写：子集/试跑绝不覆盖总账 ✓（否则 5 只的账会顶掉 5,471 只的账 ✗）
    meta_txt = "未写（子集运行，不动总账 ✓）"
    if len(loader.list_codes()) == len(codes):
        meta_txt = clean_to_disk.store_meta(meta, out_dir).name
    else:
        logger.info(f"clean 是子集运行（{len(codes)} 只）：不覆盖总账 ✓")
    return (f"{meta['stocks']} 只 / {meta['rows']:,} 行 / {meta['mb']} MB"
            f" / empty {meta['empty']} / 失败 {len(meta['failures'])}"
            f" / 总账 {meta_txt}")


def _step_panel(codes, start, end, out_dir):
    """② 面板：缓存身份会自动判定该命中还是该重建（变了就重建 ✓）"""
    from backtest.panel import load_panels        # 延迟 import：清洗层不常驻依赖回测层
    panels = load_panels()
    return f"hfq_close shape={panels['hfq_close'].shape}"


def _step_bt(codes, start, end, out_dir):
    """③ 回测：走根目录的 main.py（它自带 RUN 身份头与逐步日志 ✓）"""
    import main as backtest_main
    backtest_main.main()
    return "报告已写 reports/（含 _prev 基准）"


def run_all(codes=None, start=_ALL_START, end=_ALL_END,
            out_dir=C.CLEANER_DAILY_DIR):
    """总调度：清洗 → 面板 → 回测。

    每步一行日志（依据 + 产出）；任一步失败 ⇒ 停下并打印"后续步骤已跳过"，随后抛出。
    """
    codes = loader.list_codes() if codes is None else codes
    steps = [("clean", _step_clean), ("panel", _step_panel), ("bt", _step_bt)]
    logger.info(f"总调度开始：{len(steps)} 步 | 标的 {len(codes)} 只 | 区间 {start} ~ {end}")
    for i, (name, fn) in enumerate(steps, 1):
        logger.info(f"[{i}/{len(steps)}] {name} 开始 …")
        try:
            out = fn(codes, start, end, out_dir)
        except Exception as e:                     # 段级：直接中断，绝不"跳过继续" ✗
            left = [n for n, _ in steps[i:]]
            logger.error(f"[{i}/{len(steps)}] {name} 失败 → 后续步骤已跳过 {left}："
                         f"{type(e).__name__}: {e}")
            raise
        logger.info(f"[{i}/{len(steps)}] {name} ok 产出:{out}")
    logger.info("总调度完成 ✓（清洗 → 面板 → 回测）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        raise SystemExit(run_all())               # 总调度：python -m data.cleaner.pipeline all
    codes = loader.list_codes()[:3]
    print("候选清单前 3 只:", codes)
    run_pipeline(codes, _START, _END)
    print("跑完 ✅")