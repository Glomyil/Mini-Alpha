"""loader.py —— 数据层：parquet → DataFrame（df 的第一个生产者）

职责：选文件 → read_parquet → 区间裁剪(S0) → 守卫 → 逐只交出。
不做：排序/去重/类型修正（align 的活）、复权/指标（adjust/derived 的活）。
不进 yml 的 pipeline 列表、不写 run(df, params, ctx) 签名。

两类数据源：
  "stock"       行情日频    data/storage/stock_daily/{code}.parquet
  "fundamental" 财务期频    data/storage/fundamental_event/{code}.parquet
两者共用同一个 load_one，只有三处按 kind 微调：
  ① 必需列契约   base.check_required_columns(df, kind=kind)
  ② 时间列名     "date" / "statDate"
  ③ 区间裁剪     行情两头裁；财务【只裁上界】（见 load_one 里的注释）

迭代有两个名字，因为缺数据时语义不同：
  iter_frames  → 2 元组，行情缺 = 这票没法算，跳过
  iter_pairs   → 3 元组，财务缺 = 必须把 None 传下去（覆盖率只有 6%，不能和「票不存在」混为一谈）

用法：
    python -m data.loader                # 手动跑一只看效果
    from data.loader import list_codes, load_one, iter_frames, iter_pairs
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from config.constants import FUNDAMENTAL_EVENT_DIR, STOCK_DAILY_DIR
from data.cleaner import base

# 数据源注册表：一处登记「读哪个目录 / 哪列是时间 / 必须有哪些列」。
# 加第三类数据源只往这里加一行，loader 其余代码不用动。
# 目录常量统一放在 config/constants.py（PITFALLS #26：同一个目录不定义两遍）。
KINDS = {
    "stock": {
        "store": STOCK_DAILY_DIR,
        "time_col": "date",
        "required": base.REQUIRED_COLUMNS,
    },
    "fundamental": {
        "store": FUNDAMENTAL_EVENT_DIR,
        "time_col": "statDate",
        "required": base.FUNDAMENTAL_REQUIRED_COLUMNS,
    },
}


def _resolve(store: Path | None, kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"未知的数据类型 kind={kind!r}，可选 {sorted(KINDS)}")
    return Path(store) if store is not None else KINDS[kind]["store"]


def list_codes(store: Path | None = None, *, kind: str = "stock") -> list[str]:
    """候选清单：目录里有数据的票，含退市股不漏样本。"""
    return sorted(p.stem for p in _resolve(store, kind).glob("*.parquet"))


def load_one(code: str, start: str, end: str, store: Path | None = None,
             *, kind: str = "stock") -> Optional[pd.DataFrame]:
    """读一只并裁剪；结构错→raise，区间内无行→None。

    裁剪规则按 kind 不同：
        行情  start <= date <= end            两头裁
        财务  pubDate_max <= end               只裁上界

    财务为什么不设下界：as-of 要的是「每个交易日能看见的最新一期」，
    而 start 那天能看见的那期，statDate 可能远早于 start（比如 1 月初看到的是
    去年三季报）。按 statDate >= start 裁会把它切掉 ⇒ 凭空造出空洞，且不报错。

       参数 start 对 kind="fundamental" **不起作用** —— 保留它只为与行情签名对称。
       别以为它在财务下裁了左边界。任何在外层再补一刀 statDate >= start 的写法，
       都会把 as-of 需要的那一期切掉，而且不报错。
       （实测：2020-01-01~2020-12-31 取回 23 期而不是 4 期 —— 多出来的正是 2020 年前的期。）
    """
    store = _resolve(store, kind)
    path = store / f"{code}.parquet"
    if not path.exists():
        return None                                   # 文件不存在 = 业务现象

    df = pd.read_parquet(path)

    # 缺列 = 结构错误，必须 raise（契约按 kind 取）
    base.check_required_columns(df, KINDS[kind]["required"])

    tcol = KINDS[kind]["time_col"]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

    t1 = pd.Timestamp(end)
    if kind == "stock":
        t0 = pd.Timestamp(start)
        df = df[(df[tcol] >= t0) & (df[tcol] <= t1)]
    else:
        #    显式转一次，不依赖 fundamental_merge 那边的 CAST AS TIMESTAMP ——
        #    上游换源 / merge 没重跑时，否则这行会比出 TypeError 而不是可读的错。
        df = df.copy()
        df["pubDate_max"] = pd.to_datetime(df["pubDate_max"], errors="coerce")
        df = df[df["pubDate_max"] <= t1]              # 只裁上界，见 docstring

    if len(df) == 0:
        return None                                   # 行空 = 业务现象

    return df.reset_index(drop=True)


def iter_frames(codes: list[str], start: str, end: str, store: Path | None = None,
                *, kind: str = "stock") -> Iterator[tuple[str, pd.DataFrame]]:
    """逐只产出 (code, df)：生成器 → 内存里同时只有一只。缺数据的票跳过。"""
    for code in codes:
        df = load_one(code, start, end, store, kind=kind)
        if df is None:
            continue
        yield code, df


def iter_pairs(codes: list[str], start: str, end: str,
               store: Path | None = None, fund_store: Path | None = None,
               ) -> Iterator[tuple[str, pd.DataFrame, Optional[pd.DataFrame]]]:
    """逐只产出 (code, price_df, fund_df)。

    和 iter_frames 的差别只有一条：**财务缺失时传 None 而不是跳过** ——
    「这票没行情」= 没法算；「这票没财务」= 能算，但下游必须知道它没有。
    内部仍只调 load_one，读取逻辑一份都不重复。
    """
    for code in codes:
        price = load_one(code, start, end, store, kind="stock")
        if price is None:
            continue
        fund = load_one(code, start, end, fund_store, kind="fundamental")
        yield code, price, fund


if __name__ == "__main__":
    # 手动冒烟：覆盖 ① 主路径 ② 边界 ③ 异常分支（行空→None）④ 两种 kind
    codes = list_codes()
    print("行情候选清单:", len(codes))
    print("财务候选清单:", len(list_codes(kind="fundamental")))

    df = load_one("sh.600000", "2020-01-01", "2020-12-31")
    print("行情 2020 shape:", None if df is None else df.shape)
    assert df is not None and len(df) > 0, "主路径应当返回数据"
    assert df["date"].min() >= pd.Timestamp("2020-01-01")
    assert df["date"].max() <= pd.Timestamp("2020-12-31")

    fd = load_one("sh.600000", "2020-01-01", "2020-12-31", kind="fundamental")
    print("财务 (只裁上界) shape:", None if fd is None else fd.shape)
    if fd is not None:
        assert fd["pubDate_max"].max() <= pd.Timestamp("2020-12-31")
        assert fd["statDate"].min() < pd.Timestamp("2020-01-01"), \
            "下界被裁掉了 —— as-of 会出现空洞"

    print("区间外:", load_one("sh.600000", "2030-01-01", "2030-12-31"))
    print("不存在:", load_one("sh.999999", "2020-01-01", "2020-12-31"))
    print("冒烟通过 ✅")
