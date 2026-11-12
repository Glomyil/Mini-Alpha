"""data/index_snapshot.py —— 指数成分快照的读取器（as-of 语义）

数据从哪来
    ops/fetch_index_snapshot.py 按【自然月最后一个交易日】抓 baostock 的时点名单，
    落到 data/storage/index_snapshot/{YYYY-MM-DD}.parquet。

为什么读取必须做 as-of
    快照是【月末】的，而调仓日可能是任何交易日（freq=ME 时恰好重合，QE/W-FRI 时不然）。
    正确的取法只有一种：取"目标日及之前最近的一份"。
    取"之后最近的一份"就是前视 —— 而且它不报错，只会让回测变好看。

为什么不能静默降级
    缺文件时返回全 False，宇宙会静默缩成空池 ⇒ 权重行和为 0 ⇒ 引擎拒跑，或更糟：
    某一两层侥幸非空，于是结果看起来"跑通了"，实际样本空间是错的。
    所以缺数据一律抛错，并把"哪个指数、哪个日期、最早/最晚有哪些"写进消息。

⚠️ 本模块只做读取 + 对齐。口径怎么叠加、怎么进指纹，属于 Pool（business code）。
"""
from __future__ import annotations

import bisect
from pathlib import Path

import pandas as pd

IDX_SRC = Path("data/storage/index_snapshot")

# 可用的宇宙代号（与落盘的 bool 列名一致）
INDEX_CODES: tuple[str, ...] = ("hs300", "zz500", "zz800", "sz50")


def _snapshot_files(src: Path) -> list[Path]:
    return sorted(p for p in src.glob("*.parquet") if not p.name.startswith("_"))


def _snapshot_dates(src: Path) -> list[pd.Timestamp]:
    out = []
    for p in _snapshot_files(src):
        try:
            out.append(pd.Timestamp(p.stem))
        except ValueError as e:                        # 文件名不是日期 ⇒ 别猜，报错
            raise ValueError(f"{p} 文件名不是 YYYY-MM-DD 格式：{e}") from e
    return sorted(out)


def load_index_mask(index_code: str, target_dates, columns, *, src: Path = IDX_SRC) -> pd.DataFrame:
    """把 {index_code} 的时点名单对齐成 (target_dates × columns) 的 bool 宽表。

    as-of 语义：每个目标日取【该日及之前最近的一份】快照。
    目标日早于最早一份快照 ⇒ 抛错（不许拿未来名单回填）。
    缺文件 / 目录为空 ⇒ 抛错（不许静默变成全 False）。
    """
    if index_code not in INDEX_CODES:
        raise ValueError(f"未知指数宇宙 {index_code!r}，可用：{INDEX_CODES}")

    target_dates = [pd.Timestamp(d) for d in target_dates]
    columns = list(columns)
    src = Path(src)

    files = _snapshot_files(src)
    if not files:
        raise FileNotFoundError(
            f"指数成分快照目录为空：{src}。先跑 python -m ops.fetch_index_snapshot")
    dates = _snapshot_dates(src)
    by_date = dict(zip(dates, files))

    # 每个目标日 → 最近一份 <= 目标日的快照
    picked: list[pd.Timestamp] = []
    for d in target_dates:
        i = bisect.bisect_right(dates, d)
        if i == 0:
            raise ValueError(
                f"目标日 {d.date()} 早于最早一份快照 {dates[0].date()}（{src}）——"
                f"无法 as-of 取名单，拒绝用未来名单回填")
        picked.append(dates[i - 1])

    # 同一份文件只读一次
    cache: dict[pd.Timestamp, set] = {}
    out = pd.DataFrame(False, index=pd.DatetimeIndex(target_dates), columns=columns, dtype=bool)
    for d, snap in zip(target_dates, picked):
        if snap not in cache:
            df = pd.read_parquet(by_date[snap])
            if index_code not in df.columns:
                raise KeyError(f"{by_date[snap]} 里没有列 {index_code!r}（有：{list(df.columns)}）")
            cache[snap] = set(df.loc[df[index_code].to_numpy(dtype=bool), "code"].astype(str))
        hits = cache[snap].intersection(columns)
        if hits:
            out.loc[d, list(hits)] = True

    return out
