import json
import time
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

from data import loader
from data.cleaner.pipeline import PIPELINE
SRC = Path("data/storage/cleaner_daily")          # 净数据（已落盘，5,471 个长表）
CACHE = Path("backtest/cache")
FIELDS = ["hfq_close", "hfq_open", "ret_1d_hfq",
          "can_trade", "amount"]  # 新增宽表名单
_SELF = Path(__file__)                            # 判断"缓存是不是比这份代码新"


def build_panels(files):
    print(f"  开始重建面板：{len(files):,} 只 × {len(FIELDS)} 张表，约 30–90 秒（别中断）…")
    # """读 cleaner_daily → 4 张宽表（不做清洗、不落盘）"""
    c_l, o_l, r_l, t_l, a_l = [], [], [], [], []  # 想新建宽表就自己往等式左边加列表名，右边加列表:[]
    for p in files:
        d = pd.read_parquet(p)
        idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
        # ① hfq_close：**直接读 adjust 的产物**（唯一产出者是清洗层，panel 不再算一遍）
        c_l.append(pd.Series(pd.to_numeric(d["hfq_close"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))
        # ② hfq_open：同样直接读 adjust 的产物（方案 A，见 §2.5）
        o_l.append(pd.Series(pd.to_numeric(d["hfq_open"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))
        # ③ ret_1d_hfq：也是直接读 derived 的产物
        r_l.append(pd.Series(pd.to_numeric(d["ret_1d_hfq"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))
        ok = ((pd.to_numeric(d["is_suspended"], errors="coerce").fillna(1).to_numpy() == 0)
              & (pd.to_numeric(d["is_price_bad"], errors="coerce").fillna(1).to_numpy() == 0)
              & pd.notna(pd.to_numeric(d["hfq_close"], errors="coerce").to_numpy()))
        t_l.append(pd.Series(ok, index=idx, name=p.stem))
        # ④ amount：流动性因子用（单位只影响绝对值，不影响横截面【排名】✓）

        a_l.append(pd.Series(pd.to_numeric(d["amount"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))

        # 新增宽表格式统一使用（列表名）.append(pd.Series(pd.to_numeric(d["指标名"], errors="coerce").to_numpy(),
        #                             index=idx, name=p.stem))

    # ★ 对齐后"未上市 / 已退市"的日期才第一次出现（长表里根本没有那一行）→ 那些格子是 NaN；
    #   bool 列一进 concat 会升成 object，此时 .astype(bool) 会把 NaN 变成 True（凭空可交易）
    #   → 用 .eq(True)：只有明确写着的 True 才算可交易
    return {"hfq_close":  pd.concat(c_l, axis=1).sort_index().astype("float32"),
            "hfq_open":   pd.concat(o_l, axis=1).sort_index().astype("float32"),
            "ret_1d_hfq": pd.concat(r_l, axis=1).sort_index().astype("float32"),
            "amount":     pd.concat(a_l, axis=1).sort_index().astype("float32"),
            "can_trade":  pd.concat(t_l, axis=1).sort_index().eq(True)}
    # 在返回的字典内新增与上述格式一致的宽表："指标名":     pd.concat(a_l, axis=1).sort_index().astype("float32"),


def source_fingerprint(src=SRC):
    """只用三样：文件数 / 最新 mtime / 总大小（对净数据目录 cleaner_daily 取）"""
    files = sorted(src.glob("*.parquet"))
    return {"files": len(files),
            "max_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(max(p.stat().st_mtime for p in files))),
            "total_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1)}


def _materialize(panels: dict) -> dict:
    """Force a private, contiguous copy of every table AND verify the copy is complete.

    Why the check: an incomplete bulk copy would corrupt the panel at the source, and every
    later computation would then be a faithful execution on bad input (PITFALLS #23/#24).
    We compare the NaN pattern (cheap, catches unwritten regions) plus sampled values.
    """
    out = {}
    for k, v in panels.items():
        c = v.copy()
        a, b = v.to_numpy(), c.to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(
                f"panel copy '{k}': NaN pattern changed after copy -- incomplete copy (PITFALLS #23)")
        step = max(1, a.size // 200_000)
        if not np.array_equal(a.ravel()[::step], b.ravel()[::step], equal_nan=True):
            raise AssertionError(
                f"panel copy '{k}': sampled values differ after copy -- incomplete copy (PITFALLS #23)")
        out[k] = c
    return out


def _check_prices(panels: dict) -> None:

    c = panels["hfq_close"].to_numpy()
    o = panels["hfq_open"].to_numpy()
    ct = panels["can_trade"].to_numpy()
    z = int(((c == 0) & ct).sum() + ((o == 0) & ct).sum())
    assert z == 0, f"{z} 个格子既 0 价又标记可交易 —— 价格表/掩码表打架，拒绝载入"
    neg = int((c < 0).sum() + (o < 0).sum())
    assert neg == 0, f"{neg} 个负价格 —— 数据非法，拒绝载入"


def load_panels(start="2015-01-01", end="2026-12-31", *, src=None, cache=None, full=None):
    src, cache = Path(src or SRC), Path(cache or CACHE)      # 后三个只为测试注入
    files = sorted(src.glob("*.parquet"))
    # ① 只有"全库"才落缓存（full 是给测试注入的：None = 自动判定）
    is_full = (len(files) == len(loader.list_codes())
               ) if full is None else bool(full)
    stem = f"panel_clean_full_{start[:4]}_{end[:4]}"         # ② 名字只带口径 + 区间
    meta_p = cache / f"{stem}_meta.json"
    paths = {k: cache / f"{stem}_{k}.parquet" for k in FIELDS}

    if is_full and meta_p.exists():    # meta 是最后写的 ⇒ meta 在 = 四张表在（写序即不变量）
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if (meta["pipeline"] == [n for n, _ in PIPELINE]               # ③ 环节清单没变
                # ④ 数据还是那份
                and meta["source_fingerprint"] == source_fingerprint(src)
                and meta_p.stat().st_mtime > _SELF.stat().st_mtime):   # ⑤ 缓存比这份代码新
            panels = _materialize({k: pd.read_parquet(p)
                                   for k, p in paths.items()})
            assert panels["hfq_close"].shape == (
                meta["rows"], meta["cols"]), "缓存被改坏了"
            # 命中缓存（快路径）
            logger.info(f"panel hit shape:{panels['hfq_close'].shape}")
            _check_prices(panels)
            return panels

    panels = build_panels(files)                                        # 重建
    panels = _materialize(panels)
    if is_full:                                                          # 子集：不落盘
        cache.mkdir(parents=True, exist_ok=True)
        for k, p in paths.items():
            panels[k].to_parquet(p)
        meta_p.write_text(json.dumps({
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interval": [start, end], "n_codes": len(files),
            "pipeline": [n for n, _ in PIPELINE],
            "rows": panels["hfq_close"].shape[0], "cols": panels["hfq_close"].shape[1],
            "coverage_hfq_close": round(float(panels["hfq_close"].notna().to_numpy().mean()), 4),
            "tradable_median": int(panels["can_trade"].sum(axis=1).median()),
            "source_fingerprint": source_fingerprint(src)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    logger.info(
        f"panel build shape:{panels['hfq_close'].shape} codes:{len(files)}")
    _check_prices(panels)
    return panels


if __name__ == "__main__":
    # 全库建缓存就一条命令：python -m backtest.panel（第一次重建，第二次命中）
    t0 = time.time()
    panels = load_panels()
    used = time.time() - t0
    c = panels["hfq_close"]
    per_day = panels["can_trade"].sum(axis=1)
    size_mb = sum(p.stat().st_size for p in CACHE.glob("*.parquet")) / 2**20
    print(f"面板 {c.shape}（交易日 × 股票）    本次调用 {used:.1f}s（第一次=重建，之后=命中）")
    print(
        f"hfq_close 覆盖率 {c.notna().to_numpy().mean():.4%}    缓存 {size_mb:.1f} MB")
    print(
        f"每天可交易标的数 中位 {int(per_day.median())} / 最少 {int(per_day.min())} / 最多 {int(per_day.max())}")
