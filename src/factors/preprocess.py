# -*- coding: utf-8 -*-
"""WP2 预处理: 逐日截面 MAD-clip -> (中性化在neutralize.py) -> zscore.

顺序已确认: clip先(防离群点主导WLS) -> 回归残差 -> zscore.
一切统计量只在 groupby('date') 内估计 (泄漏清单#9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_K = 5.0  # 截面MAD截断倍数


def mad_clip(s: pd.Series, k: float = MAD_K) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return s
    lo, hi = med - k * mad, med + k * mad
    return s.clip(lo, hi)


def xsec_clip(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df.groupby("date")[cols].transform(mad_clip)
    return out


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd


def xsec_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df.groupby("date")[cols].transform(zscore)
    return out
