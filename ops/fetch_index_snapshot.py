"""ops/fetch_index_snapshot.py —— 按【自然月】抓历史时点指数成分快照

为什么必须按日抓（而不是"最新名单回填"）
    实测 2026-11-11：沪深300 在 2016-06-27 与 2020-06-29 两份名单之间换出 126 只，
    中证500 同期换出 269 只。拿最新名单回填历史 = 幸存者偏差，且是【最隐蔽】的那类
    前视（不是算错，是名单本身来自未来）。
    bs.query_hs300_stocks(date=...) 的 date 给的是【真时点快照】：
    2016-06-30 查 => 返回 300 只、updateDate=2016-06-27（都 <= 查询日）。

⚠️ 采样粒度 = 【自然月的最后一个交易日】，与调仓频率【解耦】
    指数成分调整是【日历事件】（半年一次，6/12 月），不是调仓事件。
    绑到 rebalance_dates(rule="ME") 上是错的 —— 改 yml 的 freq 会看起来像"数据缺样本"。
    月末快照对 ME / QE / W-FRI 都能用（读取侧按 as-of 取"最近一份 <= 目标日"）。

可抓的指数（baostock 能力边界）
    沪深300 (000300)  ✓
    中证500 (000905)  ✓
    上证50  (000016)  ✓
    中证800 (000906)  = 沪深300 ∪ 中证500，本脚本【自己拼】，不查
    中证1000(000852)  ✗ baostock 没有；如需用请走规则化重建并在报告标注

输出
    data/storage/index_snapshot/{YYYY-MM-DD}.parquet
        code   str    sh.600000（与面板列名同格式）
        hs300  bool
        zz500  bool
        zz800  bool   = hs300 | zz500（落盘存算好的，读取侧不做拼接）
        sz50   bool
    data/storage/index_snapshot/_provenance.parquet
        snapshot_date / index_code / effective_date / n_constituents
        （effective_date 来自 baostock 的 updateDate，用于证明"没有前视"）

用法
    python -m ops.fetch_index_snapshot              # 全量（2016-01 ~ 数据末日）
    python -m ops.fetch_index_snapshot --force      # 重抓已存在的快照
    python -m ops.fetch_index_snapshot --limit 3    # 只抓前 3 个月（调试）

断点续跑
    默认跳过已存在的快照文件 => 中途断了直接重跑同一条命令。

⚠️ 会话恢复：baostock 会偶发 10001001（用户未登录）—— 必须【重新登录后重试】，
   只重试同一条查询是没用的（行业那次的实测结论，同一套服务端行为）。

⚠️ 本脚本只负责"把原料抓到盘上"，不碰 panel / Pool / 因子 / 引擎。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import pandas as pd
from loguru import logger

OUT_DIR = Path("data/storage/index_snapshot")
PROV = OUT_DIR / "_provenance.parquet"
SLEEP_S = 0.8          # 每次查询之间的间隔（风控友好）
MAX_RETRIES = 3

# baostock 查询函数 -> 落盘列名。zz800 由这两个拼出来，故不在此表。
SOURCES = (
    ("hs300", "query_hs300_stocks", 300),
    ("zz500", "query_zz500_stocks", 500),
    ("sz50", "query_sz50_stocks", 50),
)


def _login():
    import baostock as bs
    for i in range(MAX_RETRIES):
        with contextlib.redirect_stdout(io.StringIO()):
            lg = bs.login()
        if lg.error_code == "0":
            return bs
        logger.warning(f"baostock 登录失败 {lg.error_code}: {lg.error_msg}（第 {i + 1} 次）")
        time.sleep(2.0 * (2 ** i))
    raise RuntimeError("baostock 登录连续失败，放弃")


def _is_session_error(msg) -> bool:
    """服务端把会话踢了（实测码 10001001）。重试同一条查询永远失败，必须先重连。"""
    s = str(msg)
    return "10001001" in s or "未登录" in s


def _relogin(bs):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()
    except Exception:                                  # noqa: BLE001
        pass
    return _login()


def _fetch_one_index(bs_mod, fn_name: str, date: str) -> tuple[pd.DataFrame, str]:
    """查一个指数在某日的成分。返回 (DataFrame[code, effective_date], effective_date)

    ⚠️ 不处理会话失效（交给调用方重连），这里只做纯查询重试。
    """
    last = None
    for i in range(MAX_RETRIES):
        try:
            rs = getattr(bs_mod, fn_name)(date=date)
            if rs.error_code != "0":
                raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame(columns=["code", "effective_date"]), ""
            df = pd.DataFrame(rows, columns=rs.fields)
            # baostock 返回 updateDate + code；updateDate 是这份名单的生效日
            eff = str(df["updateDate"].iloc[0])
            out = df[["code"]].copy()
            out["code"] = out["code"].astype(str).str.strip()
            out["effective_date"] = eff
            return out.drop_duplicates("code").reset_index(drop=True), eff
        except Exception as e:                         # noqa: BLE001
            last = e
            if _is_session_error(e):
                raise                              # 交给调用方重连，别在这里空转
            if i < MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** i))
    raise RuntimeError(f"{date} 抓取失败: {type(last).__name__}: {last}")


def _month_end_trading_days(index: pd.DatetimeIndex, since: int) -> list:
    """每个【自然月】的最后一个交易日。与调仓频率无关（这是日历事件）。"""
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    return [d for d in last.tolist() if pd.Timestamp(d).year >= since]


def _build_one(bs, date: str) -> tuple[pd.DataFrame, list]:
    """抓一天，拼成 code + 4 个 bool 列。所有 code 取并集后再补 False。"""
    frames = {}
    prov = []
    for col, fn_name, expect in SOURCES:
        df, eff = _fetch_one_index(bs, fn_name, date)
        if df.empty:
            logger.warning(f"{date} {col} 返回空 —— 该日该指数无成分股（数据异常）")
        else:
            n = len(df)
            if n != expect:
                # 不硬失败：官方名单偶有临时调整，但必须吵，不能静默
                logger.warning(f"{date} {col} 只数 {n} != 预期 {expect}（官方名单异常？）")
            # ⚠️ 唯一的致命检查：名单生效日不许晚于查询日（那就是前视）
            if eff and pd.Timestamp(eff) > pd.Timestamp(date):
                raise AssertionError(
                    f"{date} {col} 的名单生效日 {eff} 晚于查询日 —— baostock 返回了未来名单，拒绝落盘")
            frames[col] = set(df["code"])
            prov.append({"snapshot_date": date, "index_code": col,
                         "effective_date": eff, "n_constituents": n})
        time.sleep(SLEEP_S)

    all_codes = sorted(set().union(*frames.values())) if frames else []
    out = pd.DataFrame({"code": all_codes})
    for col in ("hs300", "zz500", "sz50"):
        s = frames.get(col, set())
        out[col] = out["code"].isin(s)
    out["zz800"] = out["hs300"] | out["zz500"]

    # 300 与 500 按编制规则应互斥；相交说明数据有问题，要吵
    both = out["hs300"] & out["zz500"]
    if bool(both.any()):
        logger.warning(f"{date} 沪深300 与 中证500 有 {int(both.sum())} 只重叠（编制规则应互斥）")

    return out, prov


def main() -> None:
    ap = argparse.ArgumentParser(description="按自然月抓历史时点指数成分快照")
    ap.add_argument("--since", type=int, default=2016, help="起始年份（与回测 since 对齐）")
    ap.add_argument("--force", action="store_true", help="重抓已存在的快照")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个月（调试）")
    args = ap.parse_args()

    from backtest.panel import load_panels
    panels = load_panels()
    dates = _month_end_trading_days(panels["hfq_close"].index, args.since)
    if args.limit:
        dates = dates[: args.limit]
    logger.info(f"指数成分快照：{len(dates)} 个月末（{dates[0].date()} ~ {dates[-1].date()}）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if args.force or not (OUT_DIR / f"{pd.Timestamp(d).date()}.parquet").exists()]
    logger.info(f"待抓 {len(todo)} 个（已存在的跳过），预计 {len(todo) * len(SOURCES) * (SLEEP_S + 0.4) / 60:.1f} 分钟")
    if not todo:
        logger.info("没有待抓的，退出")
        return

    bs = _login()
    prov_rows = []
    done = 0
    try:
        for d in todo:
            ds = str(pd.Timestamp(d).date())
            while True:
                try:
                    out, prov = _build_one(bs, ds)
                    break
                except Exception as e:                 # noqa: BLE001
                    if _is_session_error(e):
                        logger.warning(f"{ds} 会话失效，重新登录后重试")
                        bs = _relogin(bs)
                        continue
                    raise
            out.to_parquet(OUT_DIR / f"{ds}.parquet", index=False)
            prov_rows.extend(prov)
            done += 1
            if done % 10 == 0 or done == len(todo):
                logger.info(f"进度 {done}/{len(todo)}  最新 {ds}  "
                            f"hs300={int(out['hs300'].sum())} zz500={int(out['zz500'].sum())} "
                            f"zz800={int(out['zz800'].sum())} sz50={int(out['sz50'].sum())}")
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()

    # 溯源表：只追加/覆盖本次抓到的日期，保留历史
    if prov_rows:
        new = pd.DataFrame(prov_rows)
        if PROV.exists():
            old = pd.read_parquet(PROV)
            keep = old[~old["snapshot_date"].isin(set(new["snapshot_date"]))]
            new = pd.concat([keep, new], ignore_index=True)
        new.sort_values(["snapshot_date", "index_code"]).to_parquet(PROV, index=False)
        logger.info(f"溯源表已写：{PROV}（{len(new)} 行）")

    logger.info(f"完成。输出目录 {OUT_DIR}")


if __name__ == "__main__":
    main()
