# -*- coding: utf-8 -*-
"""T1.2 财务拉取 (per-code 断点续传) + attach_available_date.

接口: profit / balance / cash_flow (growth已剔除: 其同比字段用profit自建,
       且自建版能保证去年同期值也满足pubDate门控).
按 (code, year, quarter) 逐个查询, 2017Q1起.

运行: .venv/Scripts/python -m src.data.fetch_financials
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
FIN_DIR = RAW / "financials"

INTERFACES = {
    "profit": bs.query_profit_data,
    "balance": bs.query_balance_data,
    "cash_flow": bs.query_cash_flow_data,
}
YEAR0, YEAR1 = 2017, 2025
QUARTERS = (1, 2, 3, 4)


def fetch_code(code: str, year0: int, year1: int) -> dict[str, pd.DataFrame]:
    out = {}
    for name, fn in INTERFACES.items():
        frames = []
        for y in range(year0, year1 + 1):
            for q in QUARTERS:
                df = query_with_retry(fn, code=code, year=y, quarter=q)
                time.sleep(0.15)  # 节流: 控制账号级请求速率
                if len(df):
                    frames.append(df)
        out[name] = (pd.concat(frames, ignore_index=True) if frames
                     else pd.DataFrame())
    return out


def code_year_range(codes: list[str], y_pull0: int, y_pull1: int) -> dict:
    """按成分快照裁剪每股拉取年份:
    year0 = 首次在池年份-1(YoY需前置), year1 = 最后在池年份+1(覆盖退出后一期披露).
    """
    snaps = pd.read_parquet(RAW / "zz500_snapshots.parquet")
    snaps["y"] = pd.to_datetime(snaps["snapshot_date"]).dt.year
    g = snaps.groupby("code")["y"].agg(["min", "max"])
    out = {}
    for c in codes:
        if c in g.index:
            y0 = max(int(g.loc[c, "min"]) - 1, y_pull0)
            y1 = min(int(g.loc[c, "max"]) + 1, y_pull1)
        else:
            y0, y1 = y_pull0, y_pull1
        out[c] = (y0, y1)
    return out


def main():
    cfg = load_yaml(ROOT / "configs" / "universe.yaml")
    y_pull0 = int(cfg["sample"]["pull_start"][:4])
    y_pull1 = int(cfg["sample"]["end"][:4])
    codes = pd.read_csv(RAW / "universe_codes.csv")["code"].tolist()
    # 分片: python -m src.data.fetch_financials <n_shards> <shard_i>
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    shard_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    codes = codes[shard_i::n_shards]
    yr = code_year_range(codes, y_pull0, y_pull1)
    est = sum(4 * (b - a + 1) * len(INTERFACES) for a, b in yr.values())
    print(f"shard {shard_i}/{n_shards}: {len(codes)} codes, "
          f"est api calls ~{est}", flush=True)
    ckpt = Checkpoint(FIN_DIR / f"_checkpoint_{shard_i}.json")
    failed = []
    for d in INTERFACES:
        (FIN_DIR / d).mkdir(parents=True, exist_ok=True)

    with Session() as sess:
        consec = 0
        for i, code in enumerate(codes):
            if ckpt.done(code):
                continue
            for attempt in range(2):
                try:
                    y0, y1 = yr[code]
                    res = fetch_code(code, y0, y1)
                    for name, df in res.items():
                        if len(df):
                            atomic_to_parquet(df, FIN_DIR / name / f"{code}.parquet")
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
            if (i + 1) % 20 == 0:
                print(f"  progress {i+1}/{len(codes)} (done={len(ckpt)}, "
                      f"failed={len(failed)})", flush=True)
            time.sleep(0.05)

    if failed:
        pd.DataFrame(failed, columns=["code", "error"]).to_csv(
            FIN_DIR / f"_failed_{shard_i}.csv", index=False)
    print(f"done: {len(ckpt)}/{len(codes)}, failed={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
