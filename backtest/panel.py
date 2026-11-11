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
IND_SRC = Path("data/storage/industry_snapshot")   # T-6.1：按调仓日的行业快照
IND_MAP = Path("data/storage/industry_map.parquet")  # 行业字符串 → 稳定编号（跨重建不变）
_MV_WARNED = [False]   # float_mv 缺失只警告一次（5476 只票别刷屏）
_ST_WARNED = [False]   # isST 缺列也只警告一次
FIELDS = ["hfq_close", "hfq_open", "ret_1d_hfq",
          "can_trade", "amount", "reason",
          # 2026-11-04（T-6.1）：中性化所需的两张原料表
          "float_mv",   # 流通市值 = close(未复权) × volume/(turn/100)，由 cleaner/derived 产出
          "industry",   # 证监会行业编号（int16，0=未知），由 industry_snapshot 摊平而来
          # 2026-11-10：ST 状态。清洗层【故意不剔 ST】（那是股票池口径，属于研究选择，
          #   不是数据事实）；但研究层原先连 isST 都看不到 ⇒ 想剔也剔不了。补上这张表。
          "is_st"]
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


def _build_industry_wide(ref: pd.DataFrame):
    """按日快照 → 面板宽表（int16 行业编号，0 = 未知）。没有快照返回 None。

    为什么不能像其他表那样从每股 parquet 读：
        行业快照是【每个调仓日一个文件】（data/storage/industry_snapshot/{date}.parquet），
        不是每股一张时间序列。所以这里单独摊平。

    语义
        某个调仓日的快照，管到【下一个调仓日之前】的所有交易日 ⇒ 前向填充。

    编号稳定性（重要）
        行业字符串 → 编号必须【跨重建不变】，否则每次重建 id 都变，
        缓存指纹失效、跨运行的对比也断了。所以映射存盘（IND_MAP），只追加不改号。

    ⚠️ 实测依据（T-6.1 工单 §2.2）：24.1% 的股票行业变过，所以必须用【时点快照】，
       绝不允许拿最新分类回填历史。
    """
    files = sorted(IND_SRC.glob("*.parquet")) if IND_SRC.exists() else []
    if not files:
        logger.warning(f"行业快照缺失（{IND_SRC} 为空）⇒ 面板不产出 industry 表；"
                       f"中性化相关功能需等 ops/fetch_industry.py 跑完")
        return None

    # ① 读所有快照 → 行=快照日, 列=code, 值=行业字符串
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        s = d.set_index("code")["industry"].rename(pd.Timestamp(f.stem))
        frames.append(s)
    long = pd.DataFrame(frames)

    # ② 行业字符串 → 稳定编号（0 留给"未知"/空串）
    idmap = {}
    if IND_MAP.exists():
        m = pd.read_parquet(IND_MAP)
        idmap = {str(r["industry"]): int(r["id"]) for _, r in m.iterrows()}
    seen = {v for v in long.to_numpy().ravel() if isinstance(v, str) and v}
    new_vals = sorted(seen - set(idmap))
    if new_vals:
        nxt = (max(idmap.values()) + 1) if idmap else 1
        for v in new_vals:
            idmap[v] = nxt
            nxt += 1
        IND_MAP.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"industry": list(idmap), "id": list(idmap.values())}
                     ).to_parquet(IND_MAP, index=False)

    num = long.apply(lambda col: col.map(idmap).fillna(0).astype("int16"))
    num = num.reindex(columns=ref.columns)          # 对齐面板列，缺的列→NaN→0

    # ③ 对齐到交易日索引并前向填充：每个快照管到下一个快照之前
    num.index = pd.DatetimeIndex(num.index)
    out = num.reindex(ref.index, method="ffill").fillna(0).astype("int16")
    n_unknown = int((out == 0).to_numpy().mean() * 10000) / 100
    logger.info(f"industry 宽表: {out.shape} 行业数 {len(idmap)} 未知占比 {n_unknown:.2f}%")
    return out


def build_panels(files):
    print(f"  开始重建面板：{len(files):,} 只 × {len(FIELDS)} 张表，约 30–90 秒（别中断）…")
    # """读 cleaner_daily → 6 张宽表（不做清洗、不落盘）"""
    c_l, o_l, r_l, t_l, a_l, rs_l, mv_l, st_l = [], [], [], [], [], [], [], []  # 想新建宽表就自己往等式左边加列表名，右边加列表:[]
    for p in files:
        d = pd.read_parquet(p)
        # float_mv 缺失 ⇒ 给 NaN 并大声警告一次。
        #   为什么不 raise：中性化还没写（明天的活），现在报错会让整个测试套件红掉、
        #   阻塞其他工作。真正该"硬失败"的地方是明天的 neutralize()（三条恒等式会抓住 NaN）。
        #   但绝不许 fillna(0) —— 那会让"没数据"和"市值是 0"混成一件事。
        if "float_mv" not in d.columns:
            # 同上：必须下标 [0]。原来写的 if not _MV_WARNED 是死代码，
            # ⇒ float_mv 缺失的告警自 2026-11-04 加进去后一直没打过。
            if not _MV_WARNED[0]:
                logger.warning(
                    f"{p.name} 缺 float_mv 列 ⇒ 面板 float_mv 全 NaN。"
                    f"旧数据（11 列）没有 turn，需用新 KLINE_FIELDS 全量重下（T-6.1 工单 §5）")
                _MV_WARNED[0] = True
            d = d.assign(float_mv=np.nan)
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
        # ④ float_mv：同样是直接读 cleaner/derived 的产物，panel 不再算一遍
        mv_l.append(pd.Series(pd.to_numeric(d["float_mv"], errors="coerce").to_numpy(),
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
        # ④b is_st：直接读 cleaner 的 isST（清洗层已按它区分 ±5% 涨跌停，见 cleaner/limits.py）
        #    ⚠️ 同样用 float32：int 的 Series 在 concat 索引不齐时会拿 0 填缺失，
        #       而 0 在这里的意思是"不是 ST" ⇒ 会把"未上市"误判成"正常股票" ✗
        #    缺列（老文件 / 测试 fixture）一律当缺失 ⇒ NaN ⇒ 后面 .eq(1) 得 False。
        if "isST" in d.columns:
            _st = pd.to_numeric(d["isST"], errors="coerce").to_numpy()
        else:
            # ⚠️ 缺列不能静默当“全都不是 ST”：那样 include_st=false 想剔也剔不掉，而报告仍印「剔除 ST」⇒ 口径撒谎。
            #   照 float_mv 那边的做法：只警告一次，不刷屏（5476 只票）。
            # ⚠️ 必须下标取 [0]：_ST_WARNED 是非空列表，not [False] 恒为 False
            #   ⇒ 写成 if not _ST_WARNED 的话这条告警永远不会打（测试抓到的）
            if not _ST_WARNED[0]:
                logger.warning(
                    f"{p.name} 缺 isST 列 ⇒ 面板 is_st 全 False；"
                    f"include_st=false 将无效（不会剔掉任何票）")
                _ST_WARNED[0] = True
            _st = np.full(len(idx), np.nan)
        st_l.append(pd.Series(_st, index=idx, name=p.stem, dtype="float32"))

        # ⑤ amount：流动性因子用（单位只影响绝对值，不影响横截面【排名】✓）

        a_l.append(pd.Series(pd.to_numeric(d["amount"], errors="coerce").to_numpy(),
                             index=idx, name=p.stem))
        # ── 以后要加宽表，就照这三步来 ──
        #   ① 上面那行多开一个列表：c_l, o_l, r_l, t_l, a_l, rs_l, xx_l = [], [], ...
        #   ② 这里 append：xx_l.append(pd.Series(..., index=idx, name=p.stem))
        #   ③ return 字典里加一项，dtype 想清楚（⚠️ 要 fillna 的必须给 float，int 会拿 0 填缺失）
        #   ④ 最后别忘了把名字加进 FIELDS（否则缓存身份不变，不会重建）

    _close_ref = pd.concat(c_l, axis=1).sort_index()

    # ★ 对齐后"未上市 / 已退市"的日期才第一次出现（长表里根本没有那一行）→ 那些格子是 NaN；
    #   bool 列一进 concat 会升成 object，此时 .astype(bool) 会把 NaN 变成 True（凭空可交易）
    #   → 用 .eq(True)：只有明确写着的 True 才算可交易
    panels = {"hfq_close":  _close_ref.astype("float32"),
            "hfq_open":   pd.concat(o_l, axis=1).sort_index().astype("float32"),
            "ret_1d_hfq": pd.concat(r_l, axis=1).sort_index().astype("float32"),
            "amount":     pd.concat(a_l, axis=1).sort_index().astype("float32"),
            "can_trade":  pd.concat(t_l, axis=1).sort_index().eq(True),
            # reason：0=ok 1=停牌 2=涨停买不到 3=跌停卖不掉 4=不在市 5=有行无价
            # 没有这一格 ⇒ NaN ⇒ 4（不在市）。必须 fillna 后再 cast，否则留 NaN ✓
            "reason":     pd.concat(rs_l, axis=1).sort_index().fillna(4).astype("int8"),
            # 流通市值：NaN 必须留着（停牌/无 turn 日），不许 fillna(0) —— 明天取 ln 时才知道哪些要剔
            "float_mv":   pd.concat(mv_l, axis=1).sort_index().astype("float32"),
            # ST 状态（bool）。缺格一律当 False：宁可把未知当"正常股票"（与清洗层一致，
            #   清洗层也没剔 ST），也不要把"未知"当 ST 而凭空缩池 —— 后者会静默改变股票池。
            "is_st":      pd.concat(st_l, axis=1).sort_index().eq(1)}

    # 行业宽表的数据源是【按日的调仓日快照】，不是每股 parquet ⇒ 单独摊平，
    #   没有快照就跳过（T-6.1 是分步抓的，允许暂时缺失），只 warning 不报错。
    ind = _build_industry_wide(_close_ref)
    if ind is not None:
        panels["industry"] = ind

    # ── 尾部护栏：裁掉"可交易只数崩塌"的末尾若干天 ──
    #   为什么会塌：补字段（--backfill-only）沿用每只票【原有】的日期范围，
    #   所以大多数票停在同一个末日；但只要有几只票被整段重下过（或新数据只到部分票），
    #   面板的交易日索引就会被那几只拉到更晚 ⇒ 那几天全市场只剩个位数可交易
    #   ⇒ 分层为空 ⇒ 引擎的"W 行和必须 ≈1"护栏直接拒跑（实测：2026-11-02..11-09 只剩 3~4 只）
    #
    #   ⇒ 主动裁掉，别让坏尾巴流到引擎。只裁【连续尾部】，中间的空洞不动。
    #   下限用【相对值】（最大可交易只数的 5%，至少 1 只）而不是绝对数：
    #   测试 fixture 只有 3 只票，绝对下限 50 会把每一天都判死 ⇒ np.max 空数组崩。
    _n_tr = panels["can_trade"].sum(axis=1)
    _floor = max(1, int(_n_tr.max() * 0.05))
    _ok = (_n_tr >= _floor).to_numpy()
    _idx = np.nonzero(_ok)[0]
    if _idx.size == 0:
        logger.warning("面板每一天的可交易只数都低于护栏下限 ⇒ 不裁剪（数据太稀疏，检查一遍）")
    elif int(_idx[-1]) < len(_ok) - 1:
        _last = int(_idx[-1])                            # 最后一个"像样"的交易日
        _cut = panels["hfq_close"].index[_last + 1:]
        logger.warning(
            f"面板裁掉尾部 {len(_cut)} 天（可交易只数 < {_floor}，最后塌陷到 "
            f"{int(_n_tr.iloc[-1])} 只）: {str(_cut[0])[:10]} ~ {str(_cut[-1])[:10]} "
            f"⇒ 保留到 {str(panels['hfq_close'].index[_last])[:10]}")
        panels = {k: v.loc[:panels["hfq_close"].index[_last]] for k, v in panels.items()}
    return panels

    # 在返回的字典内新增与上述格式一致的宽表："指标名":     pd.concat(a_l, axis=1).sort_index().astype("float32"),


def source_fingerprint(src=SRC):
    """只用三样：文件数 / 最新 mtime / 总大小。

    ⚠️ 2026-11-04（T-6.1）：行业宽表的原料【不在】cleaner_daily，而在 industry_snapshot。
       如果不把它算进指纹，抓了新快照之后缓存不会失效 ⇒ 面板永远停在旧行业上 ✗
    """
    files = sorted(src.glob("*.parquet"))
    fp = {"files": len(files),
          "max_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(max(p.stat().st_mtime for p in files))),
          "total_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1)}
    ind = sorted(IND_SRC.glob("*.parquet")) if IND_SRC.exists() else []
    fp["ind_files"] = len(ind)
    fp["ind_max_mtime"] = (time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(max(p.stat().st_mtime for p in ind)))
                           if ind else "")
    return fp


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