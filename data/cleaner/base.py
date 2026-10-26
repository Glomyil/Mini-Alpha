import pandas as pd
import yaml as yml
REQUIRED_COLUMNS = ["date", "open", "high",
                    "low", "close", "volume"]  # 通过定义变量来保证正常运行


def check_not_empty(df, context=""):  # 定义了一个检查dataframe是否是空的函数，context用于存储文件名称
    if df is None or len(df) == 0:  # 如果里面啥都没有或是自身行数为0
        msg = "这个DataFrame是空的"
        if context:
            msg += f"({context})"  # 这里会追加报错文件的名称
        raise ValueError(msg)  # 这个时候会报错并输出提醒对应文件


# 此函数用于检查dataframe的数据是否都有在REQUIRED_COLUMNS中定义的列
def check_required_columns(df, required=None):
    # 假如required是空的，那么就会自动采用REQUIRED_COLUMNS中定义的检查清单
    cols = required if required is not None else REQUIRED_COLUMNS
    # missing列表自动基于内部的条件添加未出现的列
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺了这些必须列: {missing}")  # 此时报错并输出对应的缺失列


COLUMN_WHITELIST = {  # 在进行数据清洗时有时候会计算出新的指标要添加到原有的OHLCV数据中，为了保证数据的一致性。
    "is_suspended":     "停牌环节加：停牌=1",  # 这里先定义了一部分可以添加的指标，后续还要添加的话进行修改即可
    "hfq_close":     "复权环节加:后复权收盘价",
    "low_liquidity":  "流动性环节加：流动性差=1",
    "suspension_days": "连续停牌日期",
    "is_resume": "复牌日",
    "resume_after_days": "复牌日统计停牌日期长度",
    "is_price_bad": "价格加价格异常判定：价格不可用=1",
    "is_amt_vol_bad": "价格环节加量额判定异常：量额异常=1",
    "ret_1d_hfq": "今天相较前天复权后价格变动的百分比",
    "hfq_open":     "复权环节加：后复权开盘价（仅供比率用途 / T+1 开盘成交用）",
    "flag_open_limit_up":   "涨跌停环节加：开盘即涨停=1（买不到）",
    "flag_open_limit_down": "涨跌停环节加：开盘即跌停=1（卖不掉）",
    "flag_new_listing":     "涨跌停环节加：处于新股无涨跌幅期=1",
}


def check_new_columns(before_cols, after_cols):  # 这是一个进行清洗前后表格内对已有数据进行筛查的函数
    # new列表自动基于前一个列表的数据将后一个列表的数据进行筛选，并添加未出现的列
    new = [c for c in after_cols if c not in before_cols]
    # 这里根据上一步白名单的定义进行筛选，如出现新的指标将自动返回新添加的指标
    return [c for c in new if c not in COLUMN_WHITELIST]


def read_param(params, key, default):  # 此函数用于安全读取字典内对应的参数，如果没有或为空就输出默认值
    return params.get(key, default)  # 输出读取的值


def as_int(v, default=0):  # 此函数用于将输入的字符串或数字转换为整数，如果无法转换就会输出默认值或是报错

    try:
        return int(float(v))   # 先转 float 兼容 "1.0", 再截成 int
    except (TypeError, ValueError):
        return default


def as_float(v, default=0.0):  # 此函数用于将输入的字符串或数字转换为浮点数，如果无法转换就会默认输出并报错
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def new_step_report(name):  # 这里输出一个步骤处理的数据的报告
    return {"name": name, "status": "ok", "metrics": {}, "columns_added": [],  # 这里是报告的具体形式
            "warnings": [], "errors": []}


def add_warning(report, msg):  # 这里用于处理报告中出现的警告信息
    report["warnings"].append(msg)  # 如果出现警告就会添加信息
    if report["status"] == "ok":  # 如果状态正常就会将状态改为存在警告
        report["status"] = "has_warnings"


def add_error(report, msg, exc=None):  # 这里用于处理存在的错误信息，但不会让程序崩溃
    import traceback
    if exc:
        # 这里会输出三类数据：1.报错信息 2.报错类型 3.问题发生的位置
        msg = f"{msg} | {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    report["errors"].append(msg)  # 最后会将报错信息添加到报告中
    report["status"] = "error"  # 状态更改生效为error


def finish_step_report(report, metrics):  # 此函数用于输出函数运行后的完整的报告
    report["metrics"] = metrics
    return report

# ── 清洗环节清单（唯一权威是 pipeline.PIPELINE；这份常量给"不能依赖 pipeline 的模块"用）──
# backtest/panel.py 只需要环节名来做缓存身份校验 —— 它不该为了一串名字去 import 整个管线
# （那会让 data ↔ backtest 互相依赖）。两边一致性由 pipeline.py 的断言守着。
PIPELINE_NAMES = ("align", "suspension", "price", "limits", "adjust", "derived")