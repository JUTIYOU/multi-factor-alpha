# -*- coding: utf-8 -*-
"""WP2 中性化: 逐日 WLS 回归取残差.

- 只在当日 tradable 截面拟合(停牌/ST脏值不进回归)
- 模型: factor ~ [1, ln_mktcap] (+industry哑变量, drop_first防秩亏)
- 权重 w = sqrt(mktcap); mktcap缺失的行不进回归
- 行业固定类别表+UNKNOWN桶: 样本外不会出现未见类别
- 对非tradable行: 用当日拟合系数外推残差(保留行但不进IC的拟合集)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_design(day: pd.DataFrame, industry: bool,
                 ind_categories: list[str]) -> np.ndarray:
    parts = [np.ones(len(day)), day["ln_mktcap"].to_numpy(float)]
    if industry:
        cats = list(dict.fromkeys([*ind_categories, "UNKNOWN"]))
        ind = pd.Categorical(day["industry"].fillna("UNKNOWN"),
                             categories=cats)
        d = pd.get_dummies(ind, drop_first=True).to_numpy(float)
        parts.append(d)
    return np.column_stack(parts)


def neutralize_day(day: pd.DataFrame, factor: str, industry: bool,
                   ind_categories: list[str],
                   fit_mask: pd.Series) -> pd.Series:
    """对单日截面: fit_mask(True=tradable&可回归)上WLS, 残差= y - Xb.
    w = sqrt(mktcap). mktcap缺失 -> w=0 等效剔除: 直接不进拟合.
    """
    y = day[factor].to_numpy(float)
    X = build_design(day, industry, ind_categories)
    # WLS权重sqrt(市值): 大盘股因子值更可靠给更高权重; mktcap缺失->w=0剔除
    w = np.sqrt(day["mktcap"].clip(lower=0).fillna(0).to_numpy(float))
    fit = fit_mask.to_numpy() & np.isfinite(y) & np.isfinite(X).all(1) & (w > 0)
    resid = np.full(len(day), np.nan)
    if fit.sum() < 30:  # 截面样本不足,当日放弃
        return pd.Series(resid, index=day.index)
    Xf, yf, wf = X[fit], y[fit], w[fit]
    Xw = Xf * wf[:, None]
    beta, *_ = np.linalg.lstsq(Xw.T @ Xf, Xw.T @ yf, rcond=None)
    resid_all = y - X @ beta
    # 非拟合行也输出外推残差; y或X NaN的行 -> NaN
    valid = np.isfinite(y) & np.isfinite(X).all(1)
    resid[valid] = resid_all[valid]
    return pd.Series(resid, index=day.index)


def neutralize(df: pd.DataFrame, factors: list[str], industry: bool,
               ind_categories: list[str]) -> pd.DataFrame:
    """df需含 code,date,industry,ln_mktcap,mktcap,tradable + factor列."""
    out = df.copy()
    fit_mask = df["tradable"].fillna(False)
    for f in factors:
        out[f] = (df.groupby("date", group_keys=False)
                    .apply(lambda d: neutralize_day(
                        d, f, industry, ind_categories,
                        fit_mask.loc[d.index]), include_groups=False))
    return out
