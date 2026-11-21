# -*- coding: utf-8 -*-
"""WP2 RankIC / ICIR / 衰减.

- fwd_ret_h = close_badj(t+h)/close_badj(t)-1, shift(-h) 全日历口径(持有含停牌)
- RankIC = 逐日 spearman(factor, fwd_ret), 截面有效样本<30的日期剔除
- ICIR = mean(IC)/std(IC) * sqrt(252/h); 报告原始IC,不翻符号
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_N = 30


def add_fwd_ret(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = panel.sort_values(["code", "date"], kind="mergesort").copy()
    g = df.groupby("code")["close_badj"]
    for h in horizons:
        # shift(-h)按日历行取t+h收盘: 含停牌日的真实"买入并持有"h日收益
        df[f"fwd_ret_{h}"] = g.shift(-h) / df["close_badj"] - 1.0
    return df


def daily_ic(df: pd.DataFrame, factor: str, ret_col: str,
             mask_col: str = "tradable") -> pd.Series:
    """逐日 spearman; 只在 tradable 且两边非NaN的行上算."""
    res = {}
    for dt, d in df.groupby("date"):
        dd = d[d[mask_col].fillna(False)] if mask_col in d else d
        x, y = dd[factor], dd[ret_col]
        v = x.notna() & y.notna()
        if v.sum() < MIN_N:
            continue
        res[dt] = spearmanr(x[v], y[v]).statistic
    return pd.Series(res, name=factor)


def ic_table(df: pd.DataFrame, factors: list[str], ret_col: str,
             h: int) -> pd.DataFrame:
    """返回 factor x {ic_mean, ic_std, icir, ic_pos, n_days}."""
    rows = []
    ann = np.sqrt(252.0 / h)
    for f in factors:
        ic = daily_ic(df, f, ret_col)
        if len(ic) == 0:
            rows.append({"factor": f, "ic_mean": np.nan, "ic_std": np.nan,
                         "icir": np.nan, "ic_pos": np.nan, "n_days": 0})
            continue
        rows.append({"factor": f, "ic_mean": ic.mean(), "ic_std": ic.std(),
                     "icir": ic.mean() / ic.std() * ann if ic.std() else np.nan,
                     "ic_pos": (ic > 0).mean(), "n_days": len(ic)})
    return pd.DataFrame(rows).set_index("factor")


def ic_decay(df: pd.DataFrame, factor: str,
             horizons: list[int]) -> pd.Series:
    """同一因子在不同h的IC均值 -> 衰减曲线."""
    return pd.Series({h: daily_ic(df, factor, f"fwd_ret_{h}").mean()
                      for h in horizons}, name=factor)
