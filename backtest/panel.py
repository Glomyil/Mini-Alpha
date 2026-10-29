import json
import time
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

from data import loader
from data.cleaner.base import PIPELINE_NAMES   # 只要一串环节名，不依赖整个清洗管线
SRC = Path("data/storage/cleaner_daily")          # 净数据（已落盘，5,471 个长表）
CACHE = Path("backtest/cache")
FIELDS = ["hfq_close", "hfq_open", "ret_1d_hfq",
          "can_trade", "amount", "reason"]  # 新增宽表名单
_SELF = Path(__file__)                            # 判断"缓存是不是比这份代码新"


def _reason_of(no_susp, no_bad, has_px, lim_up, lim_dn):
    """单票层面的 reason：0=ok 1=停牌 2=涨停买不到 3=跌停卖不掉 5=有行无价。

    写入顺序 = 停牌 → 涨停 → 跌停 → 无价，【后面的覆盖前面的】：
        「优先剔除、最后分类」意味着【被剔除者的优先级最高】，
        所以 5（无价/坏价）必须【最后写】，才能盖住 1/2/3 ✓
    4（不在市）不在这里 —— 它只有装配后才知道（那格是 NaN）。
    """
    r = np.zeros(len(no_susp), dtype="int8")
    r[~no_susp] = 1                 # 停牌
    r[lim_up] = 2                   # 涨停买不到
    r[lim_dn] = 3                   # 跌停卖不掉
    r[~has_px | ~no_bad] = 5        # ★ 最后写 = 优先级最高（价格不可信，其它判据都无从谈起）
    return r


def build_panels(files):
    print(f"  开始重建面板：{len(files):,} 只 × {len(FIELDS)} 张表，约 30–90 秒（别中断）…")
    # """读 cleaner_daily → 6 张宽表（不做清洗、不落盘）"""
    c_l, o_l, r_l, t_l, a_l, rs_l = [], [], [], [], [], []  # 想新建宽表就自己往等式左边加列表名，右边加列表:[]
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
        # 五个输入各转一次 numpy，can_trade 和 reason 复用（别重复 to_numeric ✓）
        no_susp = pd.to_numeric(
            d["is_suspended"], errors="coerce").fillna(1).to_numpy() == 0
        no_bad = pd.to_numeric(
            d["is_price_bad"], errors="coerce").fillna(1).to_numpy() == 0
        has_px = pd.notna(pd.to_numeric(
            d["hfq_close"], errors="coerce").to_numpy())
        ok = no_susp & no_bad & has_px
        t_l.append(pd.Series(ok, index=idx, name=p.stem))
        # ④ reason：把"这天到底怎么了"编成一个 0–5 的数。顺序=先剔除、后分类 ✓
        rs = _reason_of(no_susp, no_bad, has_px,
                        lim_up=pd.to_numeric(d["flag_open_limit_up"],
                                             errors="coerce").fillna(0).to_numpy() == 1,
                        lim_dn=pd.to_numeric(d["flag_open_limit_down"],
                                             errors="coerce").fillna(0).to_numpy() == 1)
        # ⚠️ 必须 float32：整数 dtype 的 Series 在 concat 索引不齐时会用【0】填缺失（不是 NaN）✗
        #    ⇒ 那样 fillna(4) 就无从下手，整列会变成 0（实测踩过：sh.601268 只有 92 行却得到 2839 个 0）
        rs_l.append(pd.Series(rs, index=idx, name=p.stem, dtype="float32"))

        # ⑤ amount：流动性因子用（单位只影响绝对值，不影响横截面【排名】✓）

        a_l.append(pd.Series(pd.to_numeric(d["amount"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))
        # ── 以后要加宽表，就照这三步来 ──
        #   ① 上面那行多开一个列表：c_l, o_l, r_l, t_l, a_l, rs_l, xx_l = [], [], ...
        #   ② 这里 append：xx_l.append(pd.Series(..., index=idx, name=p.stem))
        #   ③ return 字典里加一项，dtype 想清楚（⚠️ 要 fillna 的必须给 float，int 会拿 0 填缺失）
        #   ④ 最后别忘了把名字加进 FIELDS（否则缓存身份不变，不会重建）

    # ★ 对齐后"未上市 / 已退市"的日期才第一次出现（长表里根本没有那一行）→ 那些格子是 NaN；
    #   bool 列一进 concat 会升成 object，此时 .astype(bool) 会把 NaN 变成 True（凭空可交易）
    #   → 用 .eq(True)：只有明确写着的 True 才算可交易
    return {"hfq_close":  pd.concat(c_l, axis=1).sort_index().astype("float32"),
            "hfq_open":   pd.concat(o_l, axis=1).sort_index().astype("float32"),
            "ret_1d_hfq": pd.concat(r_l, axis=1).sort_index().astype("float32"),
            "amount":     pd.concat(a_l, axis=1).sort_index().astype("float32"),
            "can_trade":  pd.concat(t_l, axis=1).sort_index().eq(True),
            # reason：0=ok 1=停牌 2=涨停买不到 3=跌停卖不掉 4=不在市 5=有行无价
            # 没有这一格 ⇒ NaN ⇒ 4（不在市）。必须 fillna 后再 cast，否则留 NaN ✓
            "reason":     pd.concat(rs_l, axis=1).sort_index().fillna(4).astype("int8")}

    # 在返回的字典内新增与上述格式一致的宽表："指标名":     pd.concat(a_l, axis=1).sort_index().astype("float32"),


def source_fingerprint(src=SRC):
    """只用三样：文件数 / 最新 mtime / 总大小（对净数据目录 cleaner_daily 取）"""
    files = sorted(src.glob("*.parquet"))
    return {"files": len(files),
            "max_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(max(p.stat().st_mtime for p in files))),
            "total_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1)}


def _materialize(panels: dict) -> dict:
    """把每张表【逼】着拷一份私有的、内存连续的副本，然后【当场验这一份是完整的】。

    为什么要验：批量拷贝如果没拷完，坏的是【源头】——
    后面所有计算都会变成"忠实地在一个坏输入上执行"，一路错到底还不报错（PITFALLS #23/#24）。

    验法：先比 NaN 的分布（便宜，专抓"没写到的区域"），再抽一些值比一比。
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
        if (meta["pipeline"] == list(PIPELINE_NAMES)               # ③ 环节清单没变
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
            "pipeline": list(PIPELINE_NAMES),
            "rows": panels["hfq_close"].shape[0], "cols": panels["hfq_close"].shape[1],
            "coverage_hfq_close": round(float(panels["hfq_close"].notna().to_numpy().mean()), 4),
            "tradable_median": int(panels["can_trade"].sum(axis=1).median()),
            "source_fingerprint": source_fingerprint(src)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    logger.info(
        f"panel build shape:{panels['hfq_close'].shape} codes:{len(files)}")
    _check_prices(panels)
    return panels


"""reason（ndarray, int8）→ can_buy / can_sell（ndarray, bool）。

    编码知识只住在这里：以后 reason 加 6，只改这两行就够，engine 一行不动。
    只吃 ndarray —— 不生成 DataFrame、不落盘、不进缓存。
    """


def derive_flags(reason):

    arr = np.asarray(reason)
    return {
        "can_buy":  ~np.isin(arr, (1, 2, 4, 5)),   # 停牌/涨停/不在市/无价 ⇒ 买不到
        "can_sell": ~np.isin(arr, (1, 3, 4, 5)),   # 停牌/跌停/不在市/无价 ⇒ 卖不掉
    }


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