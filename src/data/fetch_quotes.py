# -*- coding: utf-8 -*-
"""T1.1 行情+复权因子拉取 (per-code 断点续传).

- 日线不复权(adjustflag=3)全字段 -> data/raw/quotes/{code}.parquet
- 复权因子表 -> data/raw/adj_factor/{code}.parquet
- 中断重跑自动跳过已完成

运行: .venv/Scripts/python -m src.data.fetch_quotes
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.baostock_client import (Session, query_with_retry, Checkpoint,
                                      load_yaml, atomic_to_parquet)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
QUOTE_DIR = RAW / "quotes"
ADJ_DIR = RAW / "adj_factor"

FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
          "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST")


def fetch_one(code: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = query_with_retry(
        bs.query_history_k_data_plus, code, FIELDS,
        start_date=start, end_date=end, frequency="d", adjustflag="3")
    adj = query_with_retry(
        bs.query_adjust_factor, code=code, start_date=start, end_date=end)
    return q, adj


def main():
    cfg = load_yaml(ROOT / "configs" / "universe.yaml")
    start, end = cfg["sample"]["pull_start"], cfg["sample"]["end"]
    # 分片: python -m src.data.fetch_quotes <n_shards> <shard_i>
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    shard_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    codes = pd.read_csv(RAW / "universe_codes.csv")["code"].tolist()
    codes = codes[shard_i::n_shards]
    QUOTE_DIR.mkdir(parents=True, exist_ok=True)
    ADJ_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(QUOTE_DIR / f"_checkpoint_{shard_i}.json")
    failed = []
    print(f"shard {shard_i}/{n_shards}: {len(codes)} codes", flush=True)

    with Session() as sess:
        consec = 0
        for i, code in enumerate(codes):
            if ckpt.done(code):
                continue
            for attempt in range(2):
                try:
                    q, adj = fetch_one(code, start, end)
                    atomic_to_parquet(q, QUOTE_DIR / f"{code}.parquet")
                    if len(adj):
                        atomic_to_parquet(adj, ADJ_DIR / f"{code}.parquet")
                    ckpt.mark(code)
                    consec = 0
                    break
                except Exception as e:
                    # 会话被踢(10001001): 重登后重试一次; 仍失败记failed
                    if "10001001" in str(e) and attempt == 0 and sess.relogin():
                        continue
                    failed.append((code, str(e)))
                    print(f"  FAIL {code}: {e}", flush=True)
                    consec += 1
                    break
            if consec >= 8:
                print(f"  ABORT: {consec} consecutive failures "
                      f"(session unstable)", flush=True)
                break
            if (i + 1) % 25 == 0:
                print(f"  progress {i+1}/{len(codes)} (done={len(ckpt)}, "
                      f"failed={len(failed)})", flush=True)
            time.sleep(0.05)

    if failed:
        pd.DataFrame(failed, columns=["code", "error"]).to_csv(
            QUOTE_DIR / f"_failed_{shard_i}.csv", index=False)
    print(f"done: {len(ckpt)}/{len(codes)}, failed={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
