# -*- coding: utf-8 -*-
"""WP3 标签: fwd_ret 逐日截面rank -> [0,1].

只在 groupby('date') 内rank, 只用 tradable 行 (泄漏清单#9/11).
rank(pct=True) -> 均匀[0,1]; NaN保持NaN.
"""
from __future__ import annotations

import pandas as pd


def rank_label(df: pd.DataFrame, ret_col: str) -> pd.Series:
    """逐日 tradable 截面 pct rank. 非tradable或ret缺失 -> NaN."""
    def _r(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)
    y = (df[df["tradable"].fillna(False)]
         .groupby("date")[ret_col].transform(_r))
    return y.reindex(df.index)


def add_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"y_{h}"] = rank_label(out, f"fwd_ret_{h}")
    return out
