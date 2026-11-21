# -*- coding: utf-8 -*-
"""WP2 十分位单调性审计.

- 边界 = 训练窗(可配置)内因子值的十分位数,固定;样本外同边界
- 单调性指标: spearman(decile_id, mean_fwd_ret) 与 top-bottom价差
- 只在 tradable 截面分箱
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_BIN = 10


def train_edges(df: pd.DataFrame, factor: str,
                train_end: str | pd.Timestamp) -> np.ndarray:
    v = df.loc[(df["date"] <= train_end) & df["tradable"].fillna(False),
               factor].dropna()
    if len(v) < 100:
        return np.array([])
    return np.quantile(v, np.linspace(0, 1, N_BIN + 1))


def bin_table(df: pd.DataFrame, factor: str, ret_col: str,
              edges: np.ndarray) -> pd.DataFrame:
    """按固定边界分箱,统计每组 fwd_ret 均值/计数/胜率."""
    d = df[df["tradable"].fillna(False)][["date", factor, ret_col]].dropna()
    if len(edges) < 2 or d.empty:
        return pd.DataFrame()
    d["bin"] = pd.cut(d[factor], edges, labels=False,
                      include_lowest=True)
    g = d.groupby("bin")[ret_col]
    t = pd.DataFrame({"n": g.size(), "mean_ret": g.mean(),
                      "win_rate": g.apply(lambda s: (s > 0).mean())})
    t.index.name = "bin"
    return t.reset_index()


def monotonicity(t: pd.DataFrame) -> dict:
    if t.empty or len(t) < 5:
        return {"mono_rho": np.nan, "top_bottom": np.nan}
    rho = spearmanr(t["bin"], t["mean_ret"]).statistic
    tb = t.loc[t["bin"].idxmax(), "mean_ret"] - \
        t.loc[t["bin"].idxmin(), "mean_ret"]
    return {"mono_rho": rho, "top_bottom": tb}
