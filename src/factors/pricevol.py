# -*- coding: utf-8 -*-
"""WP2 量价因子 (factors.yaml price_volume 段).

口径(已确认):
- 滚动窗口=交易所交易日历(含停牌行,停牌日ret≈0自然计入)
- 停牌护栏: 窗口内 tradestatus==1 占比 <80% -> 该日因子NaN
- 收益口径 ret = pctChg/100; 价格水平用 *_badj 后复权
输入: panel.parquet; 输出: long表 (code,date,factor...)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GUARD = 0.8  # 停牌护栏阈值


def build(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["code", "date"], kind="mergesort").copy()
    df["ret"] = df["pctChg"] / 100.0
    # 停牌日收益按0入窗(volume=0), 护栏单列
    df["ret_w"] = df["ret"].fillna(0.0)
    df["ok"] = (df["tradestatus"] == 1).astype(float)
    g = df.groupby("code", sort=False)
    out = df[["code", "date"]].copy()

    # ---- 反转/动量 (护栏统一在文件底部 need_guard 处理) ----
    for name, w in [("rev_5", 5), ("rev_10", 10), ("rev_20", 20)]:
        out[name] = -np.expm1(g["ret_w"].transform(
            lambda s: np.log1p(s).rolling(w).sum()))
    # mom_12_1: ret(t-252..t-21)
    log1p = np.log1p(df["ret_w"])
    mom = (log1p.groupby(df["code"]).transform(lambda s: s.rolling(252).sum())
           - log1p.groupby(df["code"]).transform(lambda s: s.rolling(21).sum()))
    out["mom_12_1"] = np.expm1(mom)

    # ---- 波动/尾部 ----
    for name, w in [("vol_20", 20), ("vol_60", 60)]:
        out[name] = g["ret_w"].transform(lambda s: s.rolling(w).std())
    # ivol: ret对市场收益残差std(60) -- 用全市场等权收益代理,逐日横截面去均值
    mkt = df.groupby("date")["ret_w"].transform("mean")
    df["resid"] = df["ret_w"] - mkt
    out["ivol"] = df.groupby("code")["resid"].transform(
        lambda s: s.rolling(60).std())
    out["skew_20"] = g["ret_w"].transform(lambda s: s.rolling(20).skew())
    neg = df["ret_w"].where(df["ret_w"] < 0)
    out["down_vol"] = neg.groupby(df["code"]).transform(
        lambda s: s.rolling(20, min_periods=5).std())  # 仅下行日,>=5个
    # max_dd: 60d 最大回撤(后复权)
    cb = df["close_badj"]
    peak = cb.groupby(df["code"]).transform(lambda s: s.rolling(60).max())
    out["max_dd"] = -(peak / cb - 1.0)

    # ---- 流动性/量价 ----
    t5 = g["turn"].transform(lambda s: s.rolling(5).mean())
    t60 = g["turn"].transform(lambda s: s.rolling(60).mean())
    out["turn_mean"] = g["turn"].transform(lambda s: s.rolling(20).mean())
    out["turn_chg"] = t5 / t60 - 1.0
    out["amount_mean"] = g["amount"].transform(lambda s: s.rolling(20).mean())
    amp = (df["high"] - df["low"]) / df["preclose"]
    out["amplitude"] = amp.groupby(df["code"]).transform(
        lambda s: s.rolling(20).mean())
    v5 = g["volume"].transform(lambda s: s.rolling(5).mean())
    v60 = g["volume"].transform(lambda s: s.rolling(60).mean())
    out["vol_ratio"] = v5 / v60
    out["pv_corr"] = df.groupby("code").apply(
        lambda x: x["close"].rolling(20).corr(x["volume"]),
        include_groups=False).reset_index(level=0, drop=True)
    vpos = df["volume"].where(df["ret_w"] > 0).groupby(df["code"])
    vneg = df["volume"].where(df["ret_w"] < 0).groupby(df["code"])
    out["vol_asym"] = (vpos.transform(lambda s: s.rolling(60, min_periods=10).mean())
                       / vneg.transform(lambda s: s.rolling(60, min_periods=10).mean()) - 1.0)
    out["overnight"] = g.apply(lambda x: (x["open"] / x["preclose"] - 1.0)
                               .rolling(20).mean(), include_groups=False
                               ).reset_index(level=0, drop=True)
    intr = df["close"] / df["open"] - 1.0
    out["intraday"] = intr.groupby(df["code"]).transform(
        lambda s: s.rolling(20).mean())

    # ---- 统一停牌护栏: 窗口内 tradestatus 占比 ----
    need_guard = {"rev_5": 5, "rev_10": 10, "rev_20": 20, "mom_12_1": 252,
                  "vol_20": 20, "vol_60": 60, "ivol": 60, "skew_20": 20,
                  "down_vol": 20, "max_dd": 60, "turn_mean": 20,
                  "turn_chg": 60, "amount_mean": 20, "amplitude": 20,
                  "vol_ratio": 60, "pv_corr": 20, "vol_asym": 60,
                  "overnight": 20, "intraday": 20}
    ok = df["ok"]
    for f, w in need_guard.items():
        out[f] = out[f].where(
            ok.groupby(df["code"]).transform(
                lambda s: s.rolling(w, min_periods=w).mean() >= GUARD))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


if __name__ == "__main__":
    p = pd.read_parquet(Path(__file__).resolve().parents[2]
                        / "data" / "processed" / "panel.parquet")
    fac = build(p)
    PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
    fac.to_parquet(PROC / "factors_pricevol.parquet", index=False)
    print(f"pricevol factors: {fac.shape}, cols={fac.columns.tolist()[2:]}")
