# -*- coding: utf-8 -*-
"""T1.0/T1.3 拉取: 交易日历 / 中证500月末成分快照 / stock_basic / 行业 / 指数日线.

运行: .venv/Scripts/python -m src.data.fetch_universe
"""
from __future__ import annotations

import sys
from pathlib import Path

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.baostock_client import Session, rs_to_df, query_with_retry, load_yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def fetch_calendar(start: str, end: str) -> pd.DataFrame:
    df = query_with_retry(bs.query_trade_dates, start_date=start, end_date=end)
    df = df[df["is_trading_day"] == "1"].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["calendar_date"])
    return df[["date"]]


def fetch_zz500_snapshots(calendar: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """每个月末交易日拉一次成分 -> (date, code, updateDate) 长表(PIT)."""
    month_ends = (calendar.groupby(calendar["date"].dt.to_period("M"))["date"]
                  .max().dt.strftime("%Y-%m-%d").tolist())
    frames = []
    for d in month_ends:
        df = query_with_retry(bs.query_zz500_stocks, date=d)
        df["snapshot_date"] = pd.to_datetime(d)
        frames.append(df)
        print(f"  snapshot {d}: {len(df)} codes", flush=True)
    return pd.concat(frames, ignore_index=True)


def fetch_stock_basic() -> pd.DataFrame:
    return query_with_retry(bs.query_stock_basic)


def fetch_industry(codes: list[str]) -> pd.DataFrame:
    """逐票拉当前行业; 返回 (code, industry, updateDate)."""
    frames = []
    for i, c in enumerate(codes):
        df = query_with_retry(bs.query_stock_industry, code=c)
        frames.append(df)
        if (i + 1) % 100 == 0:
            print(f"  industry {i+1}/{len(codes)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def fetch_index(index_code: str, start: str, end: str) -> pd.DataFrame:
    df = query_with_retry(
        bs.query_history_k_data_plus, index_code,
        "date,code,open,high,low,close,volume,amount,pctChg",
        start_date=start, end_date=end, frequency="d", adjustflag="3")
    return df


def test_industry_date_param() -> bool:
    """实测 query_stock_industry 是否支持date参数(若能则行业也可PIT)."""
    try:
        rs = bs.query_stock_industry(code="sh.600000", date="2018-06-29")
        df = rs_to_df(rs)
        print(f"  industry date-param test: rows={len(df)}, "
              f"updateDate={df['updateDate'].tolist() if len(df) else 'NA'}")
        return len(df) > 0
    except Exception as e:
        print(f"  industry date-param test failed: {e}")
        return False


def main():
    cfg = load_yaml(ROOT / "configs" / "universe.yaml")
    start, end = cfg["sample"]["pull_start"], cfg["sample"]["end"]
    RAW.mkdir(parents=True, exist_ok=True)

    with Session():
        print("[1/6] 交易日历", flush=True)
        cal = fetch_calendar(start, end)
        cal.to_parquet(RAW / "trade_calendar.parquet", index=False)
        print(f"  trading days: {len(cal)}", flush=True)

        print("[2/6] 中证500月末成分快照", flush=True)
        snaps = fetch_zz500_snapshots(cal, cfg["universe"]["snapshot_freq"])
        snaps.to_parquet(RAW / "zz500_snapshots.parquet", index=False)
        codes = sorted(snaps["code"].unique())
        print(f"  union size: {len(codes)}", flush=True)
        pd.Series(codes, name="code").to_csv(RAW / "universe_codes.csv", index=False)

        print("[3/6] stock_basic", flush=True)
        basic = fetch_stock_basic()
        basic.to_parquet(RAW / "stock_basic.parquet", index=False)
        basic[basic["code"].isin(codes)].to_parquet(
            RAW / "stock_basic_universe.parquet", index=False)

        print("[4/6] 行业分类", flush=True)
        has_date = test_industry_date_param()
        ind = fetch_industry(codes)
        ind.to_parquet(RAW / "industry.parquet", index=False)
        print(f"  industry rows={len(ind)}, date_param_available={has_date}", flush=True)

        print("[5/6] 指数日线", flush=True)
        idx = fetch_index(cfg["universe"]["index_code"], start, end)
        idx.to_parquet(RAW / "index_quotes.parquet", index=False)
        print(f"  index rows={len(idx)}", flush=True)

    print("[6/6] done", flush=True)


if __name__ == "__main__":
    main()
