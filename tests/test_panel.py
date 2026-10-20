# -*- coding: utf-8 -*-
"""
tests/test_panel.py —— backtest/panel.py 验收（fixture：3 只 × 5 天，手工长表）

覆盖《教学文档/panel.py教学参考.md》§3.1 的 10 条断言 + 5 条加测（标 ➕）：

  ① 四张表形状一致 = (日期并集, 股票并集)
  ② 未上市的日期 → hfq_close 是 NaN、can_trade 是 False
  ③ 停牌日 → can_trade=False；停牌且缺价的那天 hfq_close 仍是 NaN（不许填）
  ④ is_price_bad=1 → can_trade=False（哪怕价格列有值）
  ⑤ can_trade 是真 bool；三张价格表 float32
  ⑥ index 升序唯一；columns 顺序 = 传给 build_panels 的文件顺序
  ⑦ 不丢行、不丢股票（行 = 日期并集，列 = 股票并集）
  ⑧ 写缓存 → 读回 → 与内存面板完全相等（含 dtype）
  ⑨ 指纹不变 → 命中（不重写 meta）；加一只新股 → 指纹变 → 重建（列数跟着源数据变）
  ⑩ 幂等：连跑两次装配，四张表逐格一致
  ➕ is_price_bad 是 NaN → 保守判不可交易（规格 §2.3 纪律②）
  ➕ 子集（full=False）→ 缓存目录一个文件都不落（规格 §2.4 契约①，PITFALL-014 的根）
  ➕ meta 里的环节清单对不上 → 重建，不静默用旧面板
  ➕ 缓存比 panel.py 旧（meta 的 mtime 更早）→ 重建（改了代码自动失效，不用记得删缓存）
  ➕ 缓存文件被改坏（少一行）→ 直接报错，不静默返回坏面板（规格 §4 第 3 条"为什么"）

本测试依赖的接口名字（冻结在规格 §2.6）::

    SRC / CACHE / FIELDS
    build_panels(files) -> dict[str, DataFrame]
    load_panels(start, end, *, src=None, cache=None, full=None) -> dict

fixture 走**临时目录注入**（src/cache/full 三个参数），全程不碰 backtest/cache/ 里的真缓存。

用法（项目根目录）::

    python tests/test_panel.py

期望末尾：结果: 通过 X / Y  →  全部通过 ✅ panel 装配 + 缓存可用
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import backtest.panel as panel
from data.cleaner import pipeline

passed = 0
failed = 0

START, END = "2020-01-01", "2020-12-31"


def check(cond, label):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}")


def expect_raise(fn, label, exc=AssertionError):
    global passed, failed
    try:
        fn()
    except exc as e:
        passed += 1
        print(f"  ✅ {label}（{type(e).__name__}: {str(e).splitlines()[0][:70]}）")
    except Exception as e:                          # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}：抛的是 {type(e).__name__}，不是 {exc.__name__}（{e}）")
    else:
        failed += 1
        print(f"  ❌ {label}：竟然没报错")


def frames_equal(a, b):
    """逐格 + dtype 比较；NaN 必须对 NaN（不许一边填了 0）"""
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, check_freq=False)
        return True, ""
    except AssertionError as e:
        return False, str(e).splitlines()[0].strip()[:100]


# ---------------------------------------------------------------------------
# 接口守卫：panel.py 还没写 / 名字对不上时，给一句人话，而不是一堆 traceback
# ---------------------------------------------------------------------------
_REQUIRED = ("SRC", "CACHE", "FIELDS", "build_panels", "load_panels")
_missing = [n for n in _REQUIRED if not hasattr(panel, n)]
if _missing:
    print(f"❌ backtest/panel.py 里还没有这些名字：{', '.join(_missing)}")
    print("   接口冻结在《教学文档/panel.py教学参考.md》§2.6 —— 写完再跑本测试")
    sys.exit(1)


# ---------------------------------------------------------------------------
# fixture：3 只股票 × 5 个交易日的手工长表
#
#   行格式：(日期, hfq_close, hfq_open, ret_1d_hfq, is_suspended, is_price_bad)
#
#   sh.600000  5 天全在、干净            → 5 天全可交易
#   sz.000001  01-06 停牌且缺价          → can_trade False，hfq_close 必须保持 NaN
#              01-07 停牌但价格还在      → can_trade False，hfq_close 保留 20.40（停牌不许抹价格）
#              01-08 is_price_bad = NaN  → ➕ 保守判 False
#   sz.300001  01-06 才上市（只有 3 行） → 01-02 / 01-03 = 未上市
#              01-07 is_price_bad = 1    → 价格列有值 30.50，仍必须 False
#
#   手工对账：
#     形状        (5 天, 3 只)
#     hfq_close NaN 3 格（C 的 01-02/01-03 + B 的 01-06）→ 覆盖率 12/15 = 0.80
#     每日可交易   [2, 2, 2, 1, 2]（01-07 只剩 A）→ 中位 2
# ---------------------------------------------------------------------------
DAYS = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]

ROWS = {
    "sh.600000": [
        ("2020-01-02", 10.00,  9.90, np.nan,  0, 0.0),
        ("2020-01-03", 10.20, 10.10,  0.0200, 0, 0.0),
        ("2020-01-06", 10.10, 10.20, -0.0098, 0, 0.0),
        ("2020-01-07", 10.30, 10.25,  0.0198, 0, 0.0),
        ("2020-01-08", 10.40, 10.35,  0.0097, 0, 0.0),
    ],
    "sz.000001": [
        ("2020-01-02", 20.00, 19.90, np.nan,  0, 0.0),
        ("2020-01-03", 20.40, 20.20,  0.0200, 0, 0.0),
        ("2020-01-06", np.nan, np.nan, np.nan, 1, 0.0),
        ("2020-01-07", 20.40, 20.40,  0.0000, 1, 0.0),
        ("2020-01-08", 20.60, 20.50,  0.0098, 0, np.nan),
    ],
    "sz.300001": [
        ("2020-01-06", 30.00, 29.50, np.nan,  0, 0.0),
        ("2020-01-07", 30.50, 30.00,  0.0166, 0, 1.0),
        ("2020-01-08", 30.20, 30.40, -0.0098, 0, 0.0),
    ],
    "sz.300002": [                     # ⑨ 用：后来新加的股票（验证"指纹变 → 重建"）
        ("2020-01-02", 40.00, 39.90, np.nan,  0, 0.0),
        ("2020-01-03", 40.40, 40.20,  0.0100, 0, 0.0),
        ("2020-01-06", 40.10, 40.20, -0.0074, 0, 0.0),
        ("2020-01-07", 40.30, 40.25,  0.0050, 0, 0.0),
        ("2020-01-08", 40.20, 40.35, -0.0025, 0, 0.0),
    ],
}


def write_fixture(dst, codes):
    """写 <code>.parquet 长表（只带 panel 契约里要读的列 + date）"""
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for code in codes:
        rows = ROWS[code]
        df = pd.DataFrame({
            "date":         pd.to_datetime([r[0] for r in rows]),
            "hfq_close":    [r[1] for r in rows],
            "hfq_open":     [r[2] for r in rows],
            "ret_1d_hfq":   [r[3] for r in rows],
            "is_suspended": pd.Series([r[4] for r in rows], dtype="int8"),
            # ➕ is_price_bad 用 float64：B 的 01-08 故意放 NaN，测"NaN 判不可交易"
            "is_price_bad": [r[5] for r in rows],
            # ➕ amount：panel.FIELDS 里后来加了 amount（amihud 要用），夹具必须提供
            "amount": [float(1000 * (i + 1)) for i in range(len(rows))],
        })
        # amount==0 的行必须同时不可交易，否则 engine 的"0 价/0 额可交易"断言会拦
        # 这里 amount 恒为正，所以不会触发；留一行注释说明约束即可。
        p = dst / f"{code}.parquet"
        df.to_parquet(p, index=False)
        out.append(p)
    return out


# 固定目录，不用 mkdtemp：某些受限环境不允许在"新建的随机目录"里再建子目录
# （实测 mkdtemp 返回的路径下 mkdir 会被拒绝）。固定路径哪儿都能跑，也方便看残留。
import shutil

TMP = Path(__file__).resolve().parents[1] / "logs" / "tmp_panel_fixture"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)
FX, CACHE_T = TMP / "src", TMP / "cache"
write_fixture(FX, ["sh.600000", "sz.000001", "sz.300001"])
FILES3 = sorted(FX.glob("*.parquet"))

print(f"fixture 临时目录：{TMP}")
print("── ① 形状：四张表 = (日期并集, 股票并集) ──")
P1 = panel.build_panels(FILES3)
check(set(P1) == set(panel.FIELDS), f"四张表都在：{sorted(P1)}")
check(all(v.shape == (5, 3) for v in P1.values()),
      "四张表形状一致 = (5, 3)：" + " ｜ ".join(f"{k}{v.shape}" for k, v in P1.items()))
check(list(P1["hfq_close"].index) == list(pd.to_datetime(DAYS)),
      f"index = 5 天日期并集（{DAYS[0]} … {DAYS[-1]}）")
check(list(P1["hfq_close"].columns) == ["sh.600000", "sz.000001", "sz.300001"],
      f"columns = 传给 build_panels 的文件顺序：{list(P1['hfq_close'].columns)}")

print("\n── ② 未上市：hfq_close=NaN 且 can_trade=False ──")
c, t = P1["hfq_close"], P1["can_trade"]
check(pd.isna(c.loc["2020-01-02", "sz.300001"]), "sz.300001 在 2020-01-02 未上市 → hfq_close = NaN")
check(t.loc["2020-01-02", "sz.300001"] == False, "同上 → can_trade = False")   # noqa: E712
check(int(c.loc[["2020-01-02", "2020-01-03"], "sz.300001"].isna().sum()) == 2,
      "两行未上市都是 NaN（不是 0、不是 ffill 的前值）")

print("\n── ③ 停牌：can_trade=False；缺价保持 NaN，有价不许抹 ──")
check(t.loc["2020-01-06", "sz.000001"] == False, "01-06 停牌 → can_trade = False")   # noqa: E712
check(pd.isna(c.loc["2020-01-06", "sz.000001"]), "停牌且缺价 → hfq_close 仍是 NaN（不许填）")
check(t.loc["2020-01-07", "sz.000001"] == False, "01-07 停牌 → can_trade = False")   # noqa: E712
check(c.loc["2020-01-07", "sz.000001"] == 20.40,
      f"停牌但价格还在 → hfq_close 保留 {c.loc['2020-01-07', 'sz.000001']}（掩码管得住，价格不许动）")

print("\n── ④ is_price_bad=1：价格列有值也必须 False ──")
check(t.loc["2020-01-07", "sz.300001"] == False, "01-07 价格异常 → can_trade = False")   # noqa: E712
check(c.loc["2020-01-07", "sz.300001"] == 30.50,
      f"同一天 hfq_close 仍有价 {c.loc['2020-01-07', 'sz.300001']}（掩码项是独立的一条）")

print("\n── ⑤ dtype：can_trade 真 bool，三张价格表 float32 ──")
check(P1["can_trade"].dtypes.eq(bool).all(), f"can_trade dtype = {sorted(set(map(str, P1['can_trade'].dtypes)))}")
check(all(P1[k].dtypes.eq("float32").all() for k in ("hfq_close", "hfq_open", "ret_1d_hfq")),
      "hfq_close / hfq_open / ret_1d_hfq 都是 float32")
check(int(P1["hfq_close"].isna().to_numpy().sum()) == 3,
      f"面板 NaN 格 = {int(P1['hfq_close'].isna().to_numpy().sum())} 格（C 未上市 2 + B 停牌 1）→ 一律没被填")

print("\n── ⑥ 索引与列序 ──")
check(P1["hfq_close"].index.is_monotonic_increasing and P1["hfq_close"].index.is_unique,
      "index 升序且唯一")
_rev = panel.build_panels([FILES3[1], FILES3[0]])
check(list(_rev["hfq_close"].columns) == ["sz.000001", "sh.600000"],
      f"换一个传入顺序 → columns 跟着变：{list(_rev['hfq_close'].columns)}")

print("\n── ⑦ 不丢行、不丢股票 ──")
check(all(len(v) == 5 for v in P1.values()), "行数 = 5（日期并集，不是 3 只股票的行数相加）")
check(all(v.shape[1] == 3 for v in P1.values()), "列数 = 3（3 张长表 → 3 列，一只都没丢）")

print("\n── ⑧⑨ 缓存：写 → 读回相等；命中不重建；指纹变则重建 ──")
P1 = panel.load_panels(START, END, src=FX, cache=CACHE_T, full=True)
stem = f"panel_clean_full_{START[:4]}_{END[:4]}"
meta_p = CACHE_T / f"{stem}_meta.json"
want = {k: CACHE_T / f"{stem}_{k}.parquet" for k in panel.FIELDS}
check(meta_p.exists() and all(p.exists() for p in want.values()),
      f"落盘：{stem}_<字段>.parquet ×4 + {stem}_meta.json（名字只带口径 + 区间，不带哈希）")
meta = json.loads(meta_p.read_text(encoding="utf-8"))
for k in ("built_at", "interval", "n_codes", "pipeline",
          "rows", "cols", "coverage_hfq_close", "tradable_median", "source_fingerprint"):
    if k not in meta:
        check(False, f"meta 缺字段 {k}（规格 §2.4 最小字段）")
check(meta.get("interval") == [START, END] and meta.get("pipeline") == [n for n, _ in pipeline.PIPELINE],
      f"meta.interval = {meta.get('interval')}｜pipeline = {meta.get('pipeline')}")
check(meta.get("rows") == 5 and meta.get("cols") == 3 and meta.get("n_codes") == 3,
      f"meta 记账 rows/cols/n_codes = {meta.get('rows')}/{meta.get('cols')}/{meta.get('n_codes')}")
check(meta.get("coverage_hfq_close") is not None and abs(meta["coverage_hfq_close"] - 0.8) < 1e-6,
      f"meta.coverage_hfq_close = {meta.get('coverage_hfq_close')}（12/15 = 0.80）")
check(meta.get("tradable_median") == 2,
      f"meta.tradable_median = {meta.get('tradable_median')}（每日 [2,2,2,1,2] → 2）")
check(meta.get("source_fingerprint", {}).get("files") == 3,
      f"meta.source_fingerprint = {meta.get('source_fingerprint')}")

mtime0 = meta_p.stat().st_mtime_ns
P2 = panel.load_panels(START, END, src=FX, cache=CACHE_T, full=True)
check(meta_p.stat().st_mtime_ns == mtime0, "第二次调用没重写 meta → 命中缓存")
for k in panel.FIELDS:
    ok, why = frames_equal(P1[k], P2[k])
    check(ok, f"读回 == 内存（含 dtype）：{k}{('  ' + why) if why else ''}")
_rebuild = panel.build_panels(FILES3)
for k in panel.FIELDS:
    ok, why = frames_equal(_rebuild[k], P2[k])
    check(ok, f"hit 出来的面板 == 现跑装配的面板：{k}{('  ' + why) if why else ''}")

write_fixture(FX, ["sz.300002"])                      # 源数据变了：多一只
P3 = panel.load_panels(START, END, src=FX, cache=CACHE_T, full=True)
meta2 = json.loads(meta_p.read_text(encoding="utf-8"))
check(P3["hfq_close"].shape == (5, 4), f"指纹变了 → 重建：shape {P3['hfq_close'].shape}（旧的是 (5, 3)）")
check(meta2["source_fingerprint"]["files"] == 4 and meta2["cols"] == 4,
      f"meta 也跟着重建：files = {meta2['source_fingerprint']['files']}，cols = {meta2['cols']}")
FILES4 = sorted(FX.glob("*.parquet"))

print("\n── ⑩ 幂等：连跑两次装配逐格一致 ──")
PA, PB = panel.build_panels(FILES4), panel.build_panels(FILES4)
for k in panel.FIELDS:
    ok, why = frames_equal(PA[k], PB[k])
    check(ok, f"两次装配一致：{k}{('  ' + why) if why else ''}")

print("\n── ➕ 加测（规格里写明、但不在这 10 条里的规矩）──")
check(P1["can_trade"].loc["2020-01-08", "sz.000001"] == False,      # noqa: E712
      "is_price_bad = NaN → 保守判不可交易（§2.3 纪律②）")

SUB = TMP / "cache_subset"
P4 = panel.load_panels(START, END, src=FX, cache=SUB, full=False)
check(P4["hfq_close"].shape == (5, 4), f"子集也照样装配出正确形状 {P4['hfq_close'].shape}")
check(not SUB.exists() or not list(SUB.glob("*")),
      "子集（full=False）→ 缓存目录一个文件都不落（§2.4 契约①）")

CHK = TMP / "cache_meta"
panel.load_panels(START, END, src=FX, cache=CHK, full=True)
_cmeta = CHK / f"{stem}_meta.json"
_d = json.loads(_cmeta.read_text(encoding="utf-8"))
_d["pipeline"] = ["align"]                       # 假装这份缓存是"另一条管道"写的
_cmeta.write_text(json.dumps(_d, ensure_ascii=False), encoding="utf-8")
panel.load_panels(START, END, src=FX, cache=CHK, full=True)
_d2 = json.loads(_cmeta.read_text(encoding="utf-8"))
check(_d2["pipeline"] == [n for n, _ in pipeline.PIPELINE],
      f"环节清单对不上 → 重建并把真清单写回：{_d2['pipeline']}")

_early = Path(panel.__file__).stat().st_mtime - 600        # 把 meta 的 mtime 调到"比代码旧"
os.utime(_cmeta, (_early, _early))
panel.load_panels(START, END, src=FX, cache=CHK, full=True)
check(_cmeta.stat().st_mtime > Path(panel.__file__).stat().st_mtime,
      "缓存比 panel.py 旧 → 重建（改了代码自动失效，不用记得删缓存）")

BAD = TMP / "cache_bad"
panel.load_panels(START, END, src=FX, cache=BAD, full=True)
_bad_p = BAD / f"{stem}_hfq_close.parquet"
pd.read_parquet(_bad_p).iloc[:-1].to_parquet(_bad_p)        # 少一行 = 缓存被改坏
expect_raise(lambda: panel.load_panels(START, END, src=FX, cache=BAD, full=True),
             "缓存文件被改坏（shape 与 meta 不符）→ 报错，不静默返回坏面板")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    shutil.rmtree(TMP, ignore_errors=True)
    print("全部通过 ✅ panel 装配 + 掩码 + 缓存可用（临时目录已清理）")
else:
    print(f"仍有 {failed} 项失败 ❌ 现场保留在 {TMP}（修完再删）")
    sys.exit(1)
