"""ops/fetch_industry.py —— 按【自然月】抓历史时点行业分类快照（T-6.1 的一次性脚本）

为什么必须按日抓
    实测 2026-11-04：2018-06-29 与 2026-11-09 两期之间，24.1%（871/3621）的股票行业变过。
    拿最新分类回填历史 = 把四分之一的票打错行业 = 前视偏差。
    bs.query_stock_industry(date=...) 的 date 参数给的是【真时点快照】
    （2018-06-29 查 → updateDate=2018-06-25），所以逐月抓是可行的。

⚠️ 采样粒度 = 【自然月的最后一个交易日】，与调仓频率【解耦】
    行业分类是【日历事件】（证监会定期调级），不是调仓事件。
    绑到 rebalance_dates(rule="ME") 上是错的 —— 那会让"改 yml 的 freq"看起来像"数据缺样本"。
    面板侧 _build_industry_wide() 在【全部交易日】上 ffill，所以 ME / QE / W-FRI 都能用。
    已知边界：调级最多滞后 1 个月（实测平均每年约 2% 的票调级）。

输出
    data/storage/industry_snapshot/{YYYY-MM-DD}.parquet
        code / industry（证监会大类字符串）/ industry_class（门类字母 A~S）

用法
    python -m ops.fetch_industry                # 全量（2016-01 ~ 至今）
    python -m ops.fetch_industry --force        # 重抓已存在的快照
    python -m ops.fetch_industry --limit 3      # 只抓前 3 个月（调试）

断点续跑
    默认跳过已存在的快照文件 ⇒ 中途断了直接重跑同一条命令。

⚠️ 会话恢复：baostock 会偶发 10001001（用户未登录）—— 必须【重新登录后重试】，
   只重试同一条查询是没用的（实测：不重连时剩余 47 个全部失败）。

⚠️ 本脚本只负责"把原料抓到盘上"，不碰 cleaner / panel / 因子 / 引擎。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import pandas as pd
from loguru import logger

OUT_DIR = Path("data/storage/industry_snapshot")
SLEEP_S = 0.8          # 每次查询之间的间隔（风控友好）
MAX_RETRIES = 3


def _login():
    import baostock as bs
    for i in range(MAX_RETRIES):
        with contextlib.redirect_stdout(io.StringIO()):
            lg = bs.login()
        if lg.error_code == "0":
            return bs
        logger.warning(f"baostock 登录失败 {lg.error_code}: {lg.error_msg}（第 {i+1} 次）")
        time.sleep(2.0 * (2 ** i))
    raise RuntimeError("baostock 登录连续失败，放弃")


def _is_session_error(msg) -> bool:
    """服务端把会话踢了（实测码 10001001）。这种错重试同一条查询永远失败，必须先重连。"""
    s = str(msg)
    return "10001001" in s or "未登录" in s


def _relogin(bs):
    """登出旧会话（可能已失效）再重连。登出失败无所谓，忽略。"""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()
    except Exception:                                  # noqa: BLE001
        pass
    return _login()


def fetch_one(bs, date: str) -> pd.DataFrame:
    """抓某一天的行业快照。返回 DataFrame(code, industry, industry_class)

    ⚠️ 本函数【不处理会话失效】—— 那是调用方的事（要能替换 bs 对象），
       所以这里只做纯查询重试。
    """
    import baostock as bs_mod
    last = None
    for i in range(MAX_RETRIES):
        try:
            rs = bs_mod.query_stock_industry(date=date)
            if rs.error_code != "0":
                raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame(columns=["code", "industry", "industry_class"])
            df = pd.DataFrame(rows, columns=rs.fields)[["code", "industry"]].copy()
            df["industry"] = df["industry"].astype(str).str.strip()
            # 门类字母：证监会大类是"字母+2位数字+名称"，首字符即门类（A~S 共 19 个）
            df["industry_class"] = df["industry"].str[0].where(df["industry"] != "", "")
            return df.reset_index(drop=True)
        except Exception as e:                         # noqa: BLE001
            last = e
            if _is_session_error(e):
                raise                                  # 交给调用方重连，别在这里空转
            if i < MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** i))
    raise RuntimeError(f"{date} 抓取失败: {type(last).__name__}: {last}")


def _month_end_trading_days(index: pd.DatetimeIndex, since: int) -> list:
    """每个【自然月】的最后一个交易日。与调仓频率无关（这是日历事件）。"""
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    return [d for d in last.tolist() if pd.Timestamp(d).year >= since]


def main() -> None:
    ap = argparse.ArgumentParser(description="按自然月抓历史时点行业分类快照")
    ap.add_argument("--since", type=int, default=2016, help="起始年份（与回测 since 对齐）")
    ap.add_argument("--force", action="store_true", help="重抓已存在的快照")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个月（调试）")
    args = ap.parse_args()

    from backtest.panel import load_panels
    panels = load_panels()
    dates = _month_end_trading_days(panels["hfq_close"].index, args.since)
    if args.limit:
        dates = dates[:args.limit]
    logger.info(f"industry snapshot: {len(dates)} 个自然月末 "
                f"({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bs = _login()
    ok = skipped = failed = 0
    try:
        for i, d in enumerate(dates, 1):
            d = pd.Timestamp(d)
            p = OUT_DIR / f"{d.date()}.parquet"
            if p.exists() and not args.force:
                skipped += 1
                continue
            try:
                df = fetch_one(bs, str(d.date()))
            except Exception as e:                 # noqa: BLE001
                if _is_session_error(e):
                    logger.warning(f"{d.date()} 会话失效 ⇒ 重新登录后重试")
                    try:
                        bs = _relogin(bs)
                        df = fetch_one(bs, str(d.date()))
                    except Exception as e2:        # noqa: BLE001
                        failed += 1
                        logger.error(f"{d.date()} 重连后仍失败: {e2}")
                        time.sleep(SLEEP_S)
                        continue
                else:
                    failed += 1
                    logger.error(f"{d.date()} 失败: {e}")
                    time.sleep(SLEEP_S)
                    continue
            tmp = p.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(p)
            ok += 1
            if i % 20 == 0 or i == len(dates):
                logger.info(f"{i}/{len(dates)} | ok={ok} skipped={skipped} failed={failed}")
            time.sleep(SLEEP_S * (0.5 + (i % 3) * 0.5))   # ±抖动，别做匀速机器人
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()

    logger.info(f"完成: ok={ok} skipped={skipped} failed={failed} → {OUT_DIR}")


if __name__ == "__main__":
    main()
