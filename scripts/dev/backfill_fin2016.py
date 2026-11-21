# -*- coding: utf-8 -*-
"""回填2016年财务: 年份裁剪y0=max(入池年-1,2017)裁掉了2016年报,
导致2017入池股在2017H1无已披露股本/同比因子缺2016基期.
对入池年<=2018的票补拉2016Q1-Q4三接口,与现有parquet按statDate去重合并.

运行: .venv/Scripts/python scripts/backfill_fin2016.py <n_shards> <shard_i>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.baostock_client import (Session, query_with_retry, Checkpoint,
                                      atomic_to_parquet)
from src.data.fetch_financials import INTERFACES, FIN_DIR, RAW

YEAR = 2016
QUARTERS = (1, 2, 3, 4)


def codes_need_2016() -> list[str]:
    snaps = pd.read_parquet(RAW / "zz500_snapshots.parquet")
    snaps["y"] = pd.to_datetime(snaps["snapshot_date"]).dt.year
    first = snaps.groupby("code")["y"].min()
    return sorted(first[first <= 2018].index.tolist())


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    shard_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    codes = codes_need_2016()[shard_i::n_shards]
    ckpt = Checkpoint(FIN_DIR / f"_bf2016_ckpt_{shard_i}.json")
    print(f"backfill shard {shard_i}/{n_shards}: {len(codes)} codes", flush=True)

    with Session() as sess:
        consec = 0
        for i, code in enumerate(codes):
            if ckpt.done(code):
                continue
            for attempt in range(2):
                try:
                    for name, fn in INTERFACES.items():
                        frames = []
                        for q in QUARTERS:
                            df = query_with_retry(fn, code=code, year=YEAR,
                                                  quarter=q)
                            time.sleep(0.15)
                            if len(df):
                                frames.append(df)
                        if frames:
                            new = pd.concat(frames, ignore_index=True)
                            fp = FIN_DIR / name / f"{code}.parquet"
                            if fp.exists():
                                old = pd.read_parquet(fp)
                                new = (pd.concat([old, new], ignore_index=True)
                                       .drop_duplicates("statDate",
                                                        keep="last"))
                            atomic_to_parquet(new, fp)
                    ckpt.mark(code)
                    consec = 0
                    break
                except Exception as e:
                    if "10001001" in str(e) and attempt == 0 and sess.relogin():
                        continue
                    print(f"  FAIL {code}: {e}", flush=True)
                    consec += 1
                    break
            if consec >= 8:
                print("  ABORT: consecutive failures", flush=True)
                break
            if (i + 1) % 20 == 0:
                print(f"  progress {i+1}/{len(codes)} done={len(ckpt)}",
                      flush=True)
    print(f"done: {len(ckpt)}/{len(codes)}", flush=True)


if __name__ == "__main__":
    main()
