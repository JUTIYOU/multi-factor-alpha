# -*- coding: utf-8 -*-
"""WP2 编排: 因子构建 -> 三版(原始/规模中性/行业+规模中性) -> RankIC/ICIR -> 单调性 -> 初筛表.

门禁: T1.6未过不得运行(自检data/processed/GATE_T16.ok).
运行: .venv/Scripts/python scripts/run_wp2.py [--train-end 2022-12-31]
产出:
  data/processed/factors_all.parquet (code,date,32因子原始值)
  reports/factor_neutralization.csv   (因子 x 3版 x 2h 的IC/ICIR)
  reports/factor_monotonicity.csv     (单调性审计)
  reports/factor_ic_decay.csv         (h衰减)
  reports/factor_screening.csv        (初筛表,WP3入模白名单来源)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.factors import pricevol, fundamental, preprocess, neutralize, \
    ic as icmod, binning
from src.data.baostock_client import load_yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

HORIZONS = [5, 10]
N_TRADABLE_MIN = 30


def gate_check() -> None:
    g = PROC / "GATE_T16.ok"
    if not g.exists():
        sys.exit("GATE_T16.ok 不存在: T1.6验收未通过, WP2禁止跑面板数据.")


def load_industry(panel: pd.DataFrame) -> pd.Series:
    """行业PIT: industry_history快照(query_date当available) + 当前值兜底."""
    hist_files = sorted(RAW.glob("industry_history_*.parquet"))
    cur_file = RAW / "stock_industry_current.parquet"
    cur = pd.read_parquet(cur_file)[["code", "industry"]] \
        if cur_file.exists() else pd.DataFrame(columns=["code", "industry"])
    if hist_files:
        h = pd.concat([pd.read_parquet(f) for f in hist_files],
                      ignore_index=True)
        h["query_date"] = pd.to_datetime(h["query_date"])
        # 取 query_date<=date 的最新行业
        h = h.sort_values("query_date")
        p = panel[["code", "date"]].sort_values("date")
        m = pd.merge_asof(p, h[["code", "query_date", "industry"]],
                          left_on="date", right_on="query_date", by="code",
                          direction="backward")
        ind = pd.Series(m["industry"].values, index=p.index)
        ind = ind.reindex(panel.index)
    else:
        ind = pd.Series(np.nan, index=panel.index)
    ind = ind.astype(object).fillna(
        panel["code"].map(cur.set_index("code")["industry"]))
    return ind.fillna("UNKNOWN")


def build_all(panel: pd.DataFrame) -> pd.DataFrame:
    fp = PROC / "factors_all.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    print("[wp2] build pricevol", flush=True)
    a = pricevol.build(panel)
    print("[wp2] build fundamental", flush=True)
    b = fundamental.build(panel)
    fac = a.merge(b, on=["code", "date"], how="outer")
    fac.to_parquet(fp, index=False)
    return fac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2022-12-31")
    args = ap.parse_args()
    gate_check()
    REPORTS.mkdir(exist_ok=True)

    panel = pd.read_parquet(PROC / "panel.parquet")
    if "industry" not in panel.columns:
        print("[wp2] attach industry (PIT)", flush=True)
        panel["industry"] = load_industry(panel)
    ind_cat = sorted(panel["industry"].dropna().unique().tolist())

    fac = build_all(panel)
    cfg = load_yaml(ROOT / "configs" / "factors.yaml")
    reg = [k for sec in ("price_volume", "valuation", "fundamental")
           for k, v in cfg[sec].items() if v.get("neutralize")]
    factors = [f for f in reg if f in fac.columns]
    print(f"[wp2] {len(factors)} neutralize-track factors", flush=True)

    # 合并所需列
    keep = ["code", "date", "tradable", "mktcap", "ln_mktcap", "industry",
            "close_badj"]
    df = panel[keep].merge(fac, on=["code", "date"], how="left")
    df = icmod.add_fwd_ret(df, HORIZONS)

    print("[wp2] MAD clip", flush=True)
    df = preprocess.xsec_clip(df, factors)

    # 三版: raw(clip+zscore) / size / indsize
    variants = {}
    variants["raw"] = preprocess.xsec_zscore(df, factors)
    for tag, use_ind in [("size", False), ("indsize", True)]:
        print(f"[wp2] neutralize {tag}", flush=True)
        n = neutralize.neutralize(df, factors, use_ind, ind_cat)
        variants[tag] = preprocess.xsec_zscore(n, factors)

    # IC表: 3版 x 2h
    rows = []
    for tag, vdf in variants.items():
        for h in HORIZONS:
            t = icmod.ic_table(vdf, factors, f"fwd_ret_{h}", h)
            t["variant"], t["horizon"] = tag, h
            rows.append(t.reset_index())
    ic_all = pd.concat(rows, ignore_index=True)
    ic_all.to_csv(REPORTS / "factor_neutralization.csv", index=False)

    # 衰减: 用indsize版
    dec = pd.DataFrame({f: icmod.ic_decay(variants["indsize"], f, HORIZONS)
                        for f in factors}).T
    dec.columns = [f"h{h}" for h in HORIZONS]
    dec.to_csv(REPORTS / "factor_ic_decay.csv")

    # 单调性: indsize版, 训练窗边界
    mono_rows = []
    for f in factors:
        edges = binning.train_edges(variants["indsize"], f, args.train_end)
        bt = binning.bin_table(variants["indsize"], f, "fwd_ret_5", edges)
        m = binning.monotonicity(bt)
        mono_rows.append({"factor": f, **m})
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(REPORTS / "factor_monotonicity.csv", index=False)

    # 初筛表: ic_mean(h5,indsize) |icir|>1 且方向一致(ic_pos>0.55或<0.45)
    sc = (ic_all[(ic_all["variant"] == "indsize") & (ic_all["horizon"] == 5)]
          .merge(mono, on="factor"))
    sc["selected"] = (sc["icir"].abs() > 1.0) & \
        ((sc["ic_pos"] > 0.55) | (sc["ic_pos"] < 0.45))
    sc.to_csv(REPORTS / "factor_screening.csv", index=False)
    print("[wp2] done -> reports/factor_*.csv", flush=True)


if __name__ == "__main__":
    main()
