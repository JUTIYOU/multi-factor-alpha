# -*- coding: utf-8 -*-
"""构建并落盘三版因子矩阵: raw(clip+zscore)/size/indsize(clip->中性化->zscore).
WP3-5共用输入, 只跑一次.
运行: .venv/Scripts/python scripts/build_variants.py
产出: data/processed/factors_{raw,size,indsize}.parquet (code,date,31因子)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.factors import preprocess, neutralize
from src.data.baostock_client import load_yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def load_industry(panel: pd.DataFrame) -> pd.Series:
    hist = pd.concat([pd.read_parquet(f)
                      for f in sorted(RAW.glob("industry_history_*.parquet"))],
                     ignore_index=True)
    hist["query_date"] = pd.to_datetime(hist["query_date"])
    p = panel[["code", "date"]].sort_values("date")
    m = pd.merge_asof(p, hist.sort_values("query_date")
                      [["code", "query_date", "industry"]],
                      left_on="date", right_on="query_date", by="code",
                      direction="backward")
    ind = pd.Series(m["industry"].values, index=p.index).reindex(panel.index)
    curf = RAW / "stock_industry_current.parquet"
    if curf.exists():
        cur = pd.read_parquet(curf).set_index("code")["industry"]
        ind = ind.astype(object).fillna(panel["code"].map(cur))
    return ind.fillna("UNKNOWN").astype(str)


def main() -> None:
    if not (PROC / "GATE_T16.ok").exists():
        sys.exit("gate未过")
    panel = pd.read_parquet(PROC / "panel.parquet",
                            columns=["code", "date", "tradable", "mktcap",
                                     "ln_mktcap"])
    print("[v] industry", flush=True)
    panel["industry"] = load_industry(panel)
    ind_cat = sorted(set(panel["industry"].unique()) - {"UNKNOWN"})
    fac = pd.read_parquet(PROC / "factors_all.parquet")
    cfg = load_yaml(ROOT / "configs" / "factors.yaml")
    factors = [k for sec in ("price_volume", "valuation", "fundamental")
               for k, v in cfg[sec].items() if v.get("neutralize")
               and k in fac.columns]
    df = panel.merge(fac, on=["code", "date"], how="left")
    print(f"[v] clip {len(factors)} factors", flush=True)
    df = preprocess.xsec_clip(df, factors)

    raw_z = preprocess.xsec_zscore(df, factors)
    raw_z[["code", "date", *factors]].to_parquet(
        PROC / "factors_raw.parquet", index=False)
    print("[v] raw saved", flush=True)

    for tag, use_ind in [("size", False), ("indsize", True)]:
        n = neutralize.neutralize(df, factors, use_ind, ind_cat)
        n = preprocess.xsec_zscore(n, factors)
        n[["code", "date", *factors]].to_parquet(
            PROC / f"factors_{tag}.parquet", index=False)
        print(f"[v] {tag} saved", flush=True)


if __name__ == "__main__":
    main()
