# -*- coding: utf-8 -*-
"""WP4 SHAP审计: 按P3折重训 + TreeSHAP(test折tradable样本) + 簇聚合.

- 重训与WP3完全同路径(fit_model+同一feats), 用wp3_preds验证复现性
- SHAP在test折tradable行抽样计算(解释OOS预测,非训练拟合)
- 因子按|rho|>=0.7连通聚类, 同时报告簇级合计重要性(防信用稀释)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.models import train as tr


def factor_clusters(train_df: pd.DataFrame, factors: list[str],
                    thresh: float = 0.7) -> dict[str, str]:
    """train窗tradable截面上的因子相关 -> 单链连通分量聚类."""
    d = train_df[train_df["tradable"].fillna(False)]
    d = d.sample(n=min(len(d), 100_000), random_state=0)
    corr = d[factors].corr().abs()
    parent = {f: f for f in factors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(factors):
        for b in factors[i + 1:]:
            if corr.loc[a, b] >= thresh:
                parent[find(a)] = find(b)
    return {f: find(f) for f in factors}


def shap_fold(model, test_df: pd.DataFrame, features: list[str],
              n_sample: int = 3000, seed: int = 0) -> pd.Series:
    """test折tradable样本TreeSHAP -> mean|SHAP| per factor."""
    d = test_df[test_df["tradable"].fillna(False)]
    if len(d) > n_sample:
        d = d.sample(n=n_sample, random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(d[features])
    return pd.Series(np.abs(sv).mean(axis=0), index=features)


def rank_stability(fold_ranks: pd.DataFrame) -> float:
    """各折重要性排名两两spearman的均值(fold_ranks: fold x factor)."""
    if len(fold_ranks) < 2:
        return np.nan
    cors = [spearmanr(fold_ranks.iloc[i], fold_ranks.iloc[j]).statistic
            for i in range(len(fold_ranks))
            for j in range(i + 1, len(fold_ranks))]
    return float(np.nanmean(cors))


def run_shap_audit(df: pd.DataFrame, folds: list, fold_feats: dict,
                   clusters: dict, y_col: str, h: int) -> pd.DataFrame:
    """逐折重训+SHAP -> 长表 [fold, factor, mean_abs_shap, rank]."""
    rows = []
    for i, (tr_idx, te_idx) in enumerate(folds):
        feats = fold_feats[(h, i)]
        model = tr.fit_model(df.iloc[tr_idx], feats, y_col)
        if model is None:
            continue
        imp = shap_fold(model, df.iloc[te_idx], feats, seed=i)
        r = imp.rank(ascending=False)
        for f in feats:
            rows.append({"horizon": h, "fold": i, "factor": f,
                         "mean_abs_shap": imp[f], "rank": r[f],
                         "cluster": clusters.get(f, f)})
    return pd.DataFrame(rows)


def summarize_shap(long_df: pd.DataFrame) -> pd.DataFrame:
    """折级长表 -> 因子级汇总(含簇份额)."""
    rows = []
    for (h, f), g in long_df.groupby(["horizon", "factor"]):
        piv = g.pivot_table(index="fold", values="rank",
                            aggfunc="first")
        # 折间排名稳定性: 该因子排名std + 全体两两相关(在外层算)
        rows.append({"horizon": h, "factor": f,
                     "cluster": g["cluster"].iloc[0],
                     "mean_abs_shap": g["mean_abs_shap"].mean(),
                     "mean_rank": g["rank"].mean(),
                     "rank_std": g["rank"].std(),
                     "n_folds": len(g)})
    out = pd.DataFrame(rows)
    # 簇份额: 簇内mean_abs_shap合计 / 全簇总量
    tot = out.groupby(["horizon", "cluster"])["mean_abs_shap"].transform("sum")
    out["cluster_share"] = tot / out.groupby("horizon")[
        "mean_abs_shap"].transform("sum")
    return out.sort_values(["horizon", "mean_abs_shap"], ascending=False)
