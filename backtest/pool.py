from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
# ⚠️ 必须是 from ... import dataclass：写成 import dataclasses as dataclass 导入的是【模块】，
#    拿来当装饰器用就是 TypeError: 'module' object is not callable
from loguru import logger
from data.index_snapshot import load_index_mask


@dataclass(frozen=True)
class Pool:
    can_trade: pd.DataFrame
    estimation_pool: pd.DataFrame    # 点位1：估暴露的样本
    invest_pool: pd.DataFrame        # 点位2：下注的样本
    label: str                       # banner / 指纹用

    def selectable(self) -> pd.DataFrame:
        """选股池 = 可交易 ∩ 下注池。build_target_weights 用它，engine 不用。"""
        return self.can_trade & self.invest_pool

    def benchmark_mask(self) -> pd.DataFrame:
        """基准的样本 = 实际下注范围 = 两个点位的交集。

        为什么不能只用 invest_pool：点位1 已把因子值置 NaN，更宽的
        invest_pool 传不过去 ⇒ 基准里会出现策略根本买不到的票 ⇒ 超额不可信。
        （反过来也一样：用全A基准配大盘组合，实测超额会被吞成 −0.92%。）
        """
        return self.estimation_pool & self.invest_pool


def _universe_mask(code: str, index, columns, since: int) -> pd.DataFrame:
    """把宇宙代号变成 (日期 × 股票) 的 bool 宽表。

    all ⇒ 全 True（此时【绝不】读快照文件，这样「保留全A等权」天然成立）。

    ⚠️ 热身区（研究区间之前）必须【不施加限制】，且不能把那些日期喂给 loader：
       面板从 2015-01-05 起，而快照从 2015-01-30 才有第一份
       ⇒ 整段索引喂进去会被 load_index_mask 按设计拒绝
         （"目标日早于最早一份快照 —— 拒绝用未来名单回填"）。
       热身区的因子值本来也到不了权重（调仓日都 >= since），所以放行是安全的，
       而且比"让 loader 放宽"干净：哨兵不用改。
    """
    if code == "all":
        return pd.DataFrame(True, index=index, columns=columns, dtype=bool)

    first = pd.Timestamp(f"{int(since)}-01-01")
    idx = pd.DatetimeIndex(index)
    hit = idx[idx >= first]
    if len(hit) == 0:
        raise ValueError(f"面板里没有 >= {first.date()} 的日期，无法建宇宙 {code} 的掩码")

    sub = load_index_mask(code, hit, columns)
    out = pd.DataFrame(True, index=idx, columns=list(columns), dtype=bool)   # 热身区 = 不限制
    out.loc[sub.index, :] = sub
    return out


def make_pool(panels, spec) -> Pool:
    col = panels["hfq_close"].columns
    idx = panels["hfq_close"].index
    ct = panels["can_trade"]                      # 引用，不复制

    est = _universe_mask(spec.universe_stage1, idx, col, spec.since)
    inv = _universe_mask(spec.universe_stage2, idx, col, spec.since)

    # 口径告警：两个点位相同且都非 all ⇒ 点位2 是空操作（结果等价于只开点位1）
    if spec.universe_stage1 != "all" and spec.universe_stage1 == spec.universe_stage2:
        logger.warning(
            f"universe_stage1 == universe_stage2 == {spec.universe_stage1}"
            f" ⇒ invest_pool 是空操作，结果等价于只开点位1")

    label = (
        f"estimation_pool={spec.universe_stage1} | invest_pool={spec.universe_stage2}")
    return Pool(can_trade=ct, estimation_pool=est, invest_pool=inv, label=label)
