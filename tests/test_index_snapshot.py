"""tests/test_index_snapshot.py —— 指数成分快照读取器（data/index_snapshot.py）

为什么读取器要有测试
    这份数据的价值全在【as-of 语义】上：某个调仓日只能看到"那天及之前"已生效的名单。
    取错一天就是前视，而前视不会报错、只会让回测变好看 —— 必须靠测试钉住。

覆盖
    A. as-of 选文件：恰好命中 / 落在两份之间取前一份 / 早于最早一份要报错
    B. 缺数据不许静默变成 False（否则宇宙会静默缩成空池）
    C. 形状与对齐：行=目标日、列=股票列，dtype=bool
    D. 快照自身的不变式（用真实数据）：zz800 == hs300|zz500、300∩500=∅、
       provenance 里 effective_date <= snapshot_date（没有前视）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.index_snapshot import IDX_SRC, INDEX_CODES, load_index_mask  # noqa: E402

CK = []


def ck(cond, msg):
    CK.append(bool(cond))
    print(("  \u2705 " if cond else "  \u274c ") + msg)


def _mk_src(root: Path, files: dict):
    """造一个假快照目录：{\"YYYY-MM-DD\": [code, ...]}"""
    root.mkdir(parents=True, exist_ok=True)
    for ds, codes in files.items():
        df = pd.DataFrame({"code": codes})
        for c in ("hs300", "zz500", "zz800", "sz50"):
            df[c] = c == "hs300"          # 造数据只需能区分，不需真实
        df["hs300"] = df["code"].isin(codes)
        df.to_parquet(root / f"{ds}.parquet", index=False)


TMP = Path(".tmp/idx_test_src")
if TMP.exists():
    for f in TMP.glob("*.parquet"):
        f.unlink()

# 三份快照：01-29 有 A,B；02-29 有 A,C；03-31 有 C
_mk_src(TMP, {
    "2016-01-29": ["sh.600000", "sh.600001"],
    "2016-02-29": ["sh.600000", "sh.600002"],
    "2016-03-31": ["sh.600002"],
})

print("\n\u2500\u2500 A. as-of \u9009\u6587\u4ef6\u2500\u2500")
cols = ["sh.600000", "sh.600001", "sh.600002"]
# 恰好命中快照日
m1 = load_index_mask("hs300", [pd.Timestamp("2016-01-29")], cols, src=TMP)
ck(bool(m1.loc[pd.Timestamp("2016-01-29"), "sh.600001"]),
   "\u76ee\u6807\u65e5 = \u5feb\u7167\u65e5 \u21d2 \u7528\u5f53\u65e5\u540d\u5355\uff08A,B \u5728\uff09")
ck(not bool(m1.loc[pd.Timestamp("2016-01-29"), "sh.600002"]),
   "\u76ee\u6807\u65e5 = \u5feb\u7167\u65e5 \u21d2 C \u4e0d\u5728")

# 落在两份之间 ⇒ 取【前一份】（2016-03-15 应看到 02-29 的名单 A,C）
m2 = load_index_mask("hs300", [pd.Timestamp("2016-03-15")], cols, src=TMP)
ck(bool(m2.loc[pd.Timestamp("2016-03-15"), "sh.600002"]),
   "\u5904\u4e8e\u4e24\u4efd\u4e4b\u95f4 \u21d2 \u53d6\u524d\u4e00\u4efd\uff08\u542b C\uff09")
ck(bool(m2.loc[pd.Timestamp("2016-03-15"), "sh.600000"]),
   "\u5904\u4e8e\u4e24\u4efd\u4e4b\u95f4 \u21d2 \u53d6\u524d\u4e00\u4efd\uff08\u542b A\uff09")
ck(not bool(m2.loc[pd.Timestamp("2016-03-15"), "sh.600001"]),
   "\u4e0d\u53d6\u540e\u4e00\u4efd\uff08B \u5df2\u88ab 02-29 \u5254\u9664\uff0c\u4e0d\u80fd\u56de\u6d41\uff09")

# 早于最早一份 ⇒ 必须报错（否则就是拿未来名单回填）
try:
    load_index_mask("hs300", [pd.Timestamp("2015-12-31")], cols, src=TMP)
    ck(False, "\u65e9\u4e8e\u6700\u65e9\u5feb\u7167 \u21d2 \u7adf\u7136\u6ca1\u62a5\u9519")
except Exception as e:
    ck(True, f"\u65e9\u4e8e\u6700\u65e9\u5feb\u7167 \u21d2 {type(e).__name__}\uff08\u4e0d\u5141\u8bb8\u62ff\u672a\u6765\u540d\u5355\uff09")

print("\n\u2500\u2500 B. \u7f3a\u6570\u636e\u4e0d\u8bb8\u9759\u9ed8\u2500\u2500")
EMPTY = Path(".tmp/idx_test_empty")
EMPTY.mkdir(parents=True, exist_ok=True)
for f in EMPTY.glob("*.parquet"):
    f.unlink()
try:
    load_index_mask("hs300", [pd.Timestamp("2016-01-29")], cols, src=EMPTY)
    ck(False, "\u76ee\u5f55\u5168\u7a7a \u21d2 \u7adf\u7136\u4e0d\u62a5\u9519")
except Exception as e:
    ck(True, f"\u76ee\u5f55\u5168\u7a7a \u21d2 {type(e).__name__}\uff08\u4e0d\u80fd\u9759\u9ed8\u53d8\u7a7a\u6c60\uff09")

print("\n\u2500\u2500 C. \u5f62\u72b6\u4e0e\u5bf9\u9f50\u2500\u2500")
m3 = load_index_mask("hs300", [pd.Timestamp("2016-01-29"), pd.Timestamp("2016-03-31")], cols, src=TMP)
ck(m3.shape == (2, 3), f"shape \u5e94\u4e3a (2, 3)\uff08\u5b9e\u6d4b {m3.shape}\uff09")
ck(m3.dtypes.eq(bool).all(), "dtype \u5fc5\u987b\u5168 bool")
ck(list(m3.index) == [pd.Timestamp("2016-01-29"), pd.Timestamp("2016-03-31")], "\u884c\u7d22\u5f15 = \u76ee\u6807\u65e5\uff08\u987a\u5e8f\u4e0d\u53d8\uff09")
ck(list(m3.columns) == cols, "\u5217 = \u4f20\u5165\u7684\u80a1\u7968\u5217")

print("\n\u2500\u2500 D. \u771f\u5b9e\u5feb\u7167\u7684\u4e0d\u53d8\u5f0f\u2500\u2500")
files = sorted(f for f in IDX_SRC.glob("*.parquet") if not f.name.startswith("_"))
if not files:
    print("  \u26a0 \u771f\u5b9e\u5feb\u7167\u8fd8\u6ca1\u843d\u76d8\uff0c\u8df3\u8fc7 D")
else:
    d = pd.read_parquet(files[0])
    ck(bool((d["zz800"] == (d["hs300"] | d["zz500"])).all()), "zz800 \u6046\u7b49\u4e8e hs300 | zz500")
    ck(int((d["hs300"] & d["zz500"]).sum()) == 0, "\u6caa\u6df1300 \u4e0e \u4e2d\u8bc1500 \u5e94\u4e92\u65a5")
    ck(set(INDEX_CODES) >= {"hs300", "zz500", "zz800"}, f"INDEX_CODES \u8986\u76d6\u57fa\u7840\u5e94\u7528\u96c6\uff1a{INDEX_CODES}")
    prov = pd.read_parquet(IDX_SRC / "_provenance.parquet")
    lag = (pd.to_datetime(prov["snapshot_date"]) - pd.to_datetime(prov["effective_date"])).dt.days
    ck(bool((lag >= 0).all()), f"\u540d\u5355\u751f\u6548\u65e5 <= \u5feb\u7167\u65e5\uff08\u6700\u957f\u6ede\u540e {int(lag.max())} \u5929\uff09\u21d2 \u65e0\u524d\u89c6")
    ck(int(prov["n_constituents"].min()) >= 50, "\u6bcf\u4efd\u540d\u5355\u53ea\u6570\u4e0d\u5f97\u5c0f\u4e8e 50")

passed = sum(1 for c in CK if c)
print("\n" + "=" * 46)
print("  \u7ed3\u679c: \u901a\u8fc7 %d / %d" % (passed, len(CK)))
if passed != len(CK):
    print("  \u4ecd\u6709 %d \u9879\u5931\u8d25 \u274c" % (len(CK) - passed))
    sys.exit(1)
