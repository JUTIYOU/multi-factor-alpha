# -*- coding: utf-8 -*-
"""T1.3补充 行业历史(PIT): date参数实测有效, 按半年度(6/12月末)网格拉取.

输出 data/raw/industry_history.parquet: (code, query_date, industry, updateDate)
用法: industry_at(t) = max(query_date<=t) 的industry.

运行: .venv/Scripts/python -m src.data.fetch_industry_history
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.baostock_client import (Session, query_with_retry, Checkpoint,
                                      atomic_to_parquet)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def main():
    cal = pd.read_parquet(RAW / "trade_calendar.parquet")
    cal["ym"] = cal["date"].dt.to_period("M")
    # 每年6月/12月的最后一个交易日 + 最新一日
    grid = (cal[cal["ym"].dt.month.isin([6, 12])]
            .groupby("ym")["date"].max().tolist())
    last = cal["date"].max()
    if last not in grid:
        grid.append(last)
    codes = pd.read_csv(RAW / "universe_codes.csv")["code"].tolist()
    # 分片: python -m src.data.fetch_industry_history <n_shards> <shard_i>
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    shard_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    codes = codes[shard_i::n_shards]
    ckpt = Checkpoint(RAW / "industry_history_ckpt.json")
    out_path = RAW / f"industry_history_{shard_i}.parquet"

    frames = []
    if out_path.exists():
        frames.append(pd.read_parquet(out_path))
    todo = [(c, d) for c in codes for d in grid
            if not ckpt.done(f"{c}|{d:%Y-%m-%d}")]
    print(f"shard {shard_i}/{n_shards}: grid={len(grid)} x {len(codes)} "
          f"codes, todo={len(todo)}", flush=True)

    with Session() as sess:
        consec = 0
        for i, (c, d) in enumerate(todo):
            for attempt in range(2):
                try:
                    df = query_with_retry(bs.query_stock_industry,
                                          code=c, date=f"{d:%Y-%m-%d}")
                    if len(df):
                        df["query_date"] = d
                        frames.append(df[["code", "industry", "updateDate",
                                          "query_date"]])
                    ckpt.mark(f"{c}|{d:%Y-%m-%d}")
                    consec = 0
                    break
                except Exception as e:
                    if "10001001" in str(e) and attempt == 0 and sess.relogin():
                        continue
                    print(f"  FAIL {c} {d:%Y-%m-%d}: {e}", flush=True)
                    consec += 1
                    break
            if consec >= 20:
                print(f"  ABORT: {consec} consecutive failures", flush=True)
                break
            if (i + 1) % 2000 == 0:
                print(f"  progress {i+1}/{len(todo)}", flush=True)
            time.sleep(0.25)  # 节流: 控制账号级请求速率
            if len(frames) > 200:
                merged = pd.concat(frames, ignore_index=True)
                atomic_to_parquet(merged, out_path)
                frames = [merged]

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(["code", "query_date"],
                                        keep="last")
        atomic_to_parquet(merged, out_path)
    print(f"done: ckpt={len(ckpt)}, rows~={len(frames[0]) if frames else 0}",
          flush=True)


if __name__ == "__main__":
    main()
