# -*- coding: utf-8 -*-
"""WP3 评估: 复用 factors/ic.py 口径 + 十分位价差.

逐日RankIC/ICIR/ic_pos + 预测分十分位 top-bottom 平均fwd_ret.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def eval_signal(df: pd.DataFrame, pred_col: str, ret_col: str,
                n_bins: int = 10) -> dict:
    """df须含 date/tradable/pred_col/ret_col. 只在tradable截面评估."""
    d = df[df["tradable"].fillna(False)
           & df[pred_col].notna() & df[ret_col].notna()]
    ic = d.groupby("date").apply(
        lambda g: g[pred_col].corr(g[ret_col], method="spearman")
        if len(g) > 20 else np.nan, include_groups=False).dropna()
    # 十分位top-bottom价差
    def _spread(g: pd.DataFrame) -> float:
        if len(g) < n_bins * 5:
            return np.nan
        q = pd.qcut(g[pred_col], n_bins, labels=False, duplicates="drop")
        if q.nunique() < n_bins:
            return np.nan
        return float(g[ret_col].groupby(q).mean().iloc[-1]
                     - g[ret_col].groupby(q).mean().iloc[0])
    sp = d.groupby("date").apply(_spread, include_groups=False).dropna()
    return {
        "ic_mean": ic.mean(), "ic_std": ic.std(),
        "icir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "ic_pos": (ic > 0).mean(), "n_days": len(ic),
        "top_bottom": sp.mean(), "spread_std": sp.std(),
    }
