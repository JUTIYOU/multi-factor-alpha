# -*- coding: utf-8 -*-
"""行业映射持久化: merge_asof历史快照+当前兜底 -> industry_map.parquet.
WP5/WP6共用, 避免各处重复join.
运行: .venv/Scripts/python scripts/build_industry_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main() -> None:
    panel = pd.read_parquet(PROC / "panel.parquet", columns=["code", "date"])
    hist = pd.concat([pd.read_parquet(f)
                      for f in sorted(RAW.glob("industry_history_*.parquet"))],
                     ignore_index=True)
    hist["query_date"] = pd.to_datetime(hist["query_date"])
    p = panel.sort_values("date")
    m = pd.merge_asof(p, hist.sort_values("query_date")
                      [["code", "query_date", "industry"]],
                      left_on="date", right_on="query_date", by="code",
                      direction="backward")
    ind = pd.Series(m["industry"].values, index=p.index).reindex(panel.index)
    curf = RAW / "stock_industry_current.parquet"
    if curf.exists():
        cur = pd.read_parquet(curf).set_index("code")["industry"]
        ind = ind.astype(object).fillna(panel["code"].map(cur))
    out = panel.assign(industry=ind.fillna("UNKNOWN").astype(str))
    out.to_parquet(PROC / "industry_map.parquet", index=False)
    print(f"[ind] saved {len(out)} rows, "
          f"{out.industry.nunique()} industries, "
          f"UNKNOWN share={(out.industry == 'UNKNOWN').mean():.3f}")


if __name__ == "__main__":
    main()
