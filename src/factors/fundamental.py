# -*- coding: utf-8 -*-
"""WP2 估值+基本面因子.

- 估值4因子 = 行情接口 TTM 估值倒数 (T1.6 与自建门控BP对拍后启用)
- 基本面8因子: profit/balance/cash_flow 三表 -> attach_available_date -> asof_join
- g_rev/g_np 同比: 最新披露statDate值 / 去年同期statDate值 - 1;
  去年同期记录同样要求 available_date<=t (双重PIT校验)
- growth接口不拉,YOY由profit.MBRevenue/netProfit自建(注册表备注)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.pit import attach_available_date, asof_join

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

VALUATION = {"BP": "pbMRQ", "EP": "peTTM", "SP": "psTTM", "CFP": "pcfNcfTTM"}
PROFIT_FIELDS = ["roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM",
                 "MBRevenue"]
BAL_FIELDS = ["liabilityToAsset"]
CF_FIELDS = ["CFOToNP"]


def _load_endpoint(name: str) -> pd.DataFrame:
    files = sorted((RAW / "financials" / name).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in df.columns:
        if c not in ("code", "pubDate", "statDate"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _yoy(panel_fin: pd.DataFrame, fin: pd.DataFrame, value_col: str,
         suffix: str) -> pd.Series:
    """最新披露值 / 去年同期披露值 - 1, 两期都满足 available<=t."""
    cur = panel_fin[["code", "date", f"{suffix}statDate",
                     f"{suffix}{value_col}"]].copy()
    cur["prev_stat"] = cur[f"{suffix}statDate"] - pd.DateOffset(years=1)
    prev = fin[["code", "statDate", "available_date", value_col]].rename(
        columns={"statDate": "prev_stat", value_col: "_prev_val"})
    m = cur.merge(prev, on=["code", "prev_stat"], how="left")
    ok = m["available_date"].notna() & (m["available_date"] <= m["date"])
    r = m[f"{suffix}{value_col}"] / m["_prev_val"].where(ok) - 1.0
    return r


def build(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["code", "date"], kind="mergesort")
    out = p[["code", "date"]].copy()

    # 估值: TTM估值为0/负时倒数符号保留(EP<0即亏损), inf->NaN
    for name, col in VALUATION.items():
        out[name] = (1.0 / p[col].where(p[col] != 0)).replace(
            [np.inf, -np.inf], np.nan)

    fin_p = attach_available_date(_load_endpoint("profit")
                                  .dropna(subset=["statDate"]))
    fin_b = attach_available_date(_load_endpoint("balance")
                                  .dropna(subset=["statDate"]))
    fin_c = attach_available_date(_load_endpoint("cash_flow")
                                  .dropna(subset=["statDate"]))
    if fin_p.empty:
        return out

    m = asof_join(p[["code", "date"]], fin_p, PROFIT_FIELDS, suffix="fp_")
    m = m.rename(columns={"_fin_statDate": "fp_statDate"})
    mb = asof_join(p[["code", "date"]], fin_b, BAL_FIELDS, suffix="fb_")
    mc = asof_join(p[["code", "date"]], fin_c, CF_FIELDS, suffix="fc_")
    m = m.merge(mb[["code", "date", "fb_liabilityToAsset"]],
                on=["code", "date"], how="left")
    m = m.merge(mc[["code", "date", "fc_CFOToNP"]],
                on=["code", "date"], how="left")

    out["roe"] = m["fp_roeAvg"]
    out["gpm"] = m["fp_gpMargin"]
    out["npm"] = m["fp_npMargin"]
    out["lev"] = -m["fb_liabilityToAsset"]
    out["cfo_np"] = m["fc_CFOToNP"]
    out["eps_ttm"] = m["fp_epsTTM"]
    out["g_rev"] = _yoy(m, fin_p, "MBRevenue", "fp_")
    out["g_np"] = _yoy(m, fin_p, "netProfit", "fp_")
    return out.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    p = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    fac = build(p)
    PROC = ROOT / "data" / "processed"
    fac.to_parquet(PROC / "factors_fundamental.parquet", index=False)
    print(f"fundamental factors: {fac.shape}")
