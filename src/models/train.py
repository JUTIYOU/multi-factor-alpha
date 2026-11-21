# -*- coding: utf-8 -*-
"""WP3 训练: LightGBM 回归统一接口.

- 逐日IC矩阵对全frame只算一次; 每折切片取|ICIR|top-K(嵌套筛选,防selection bias)
- train内部按时间切尾20%做early-stop验证集
- 超参固定不调参(防调参选择偏差)
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

LGBM_PARAMS = dict(
    num_leaves=31, learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
    n_jobs=-1, verbosity=-1,
)


def daily_ic_matrix(df: pd.DataFrame, factors: list[str],
                    y_col: str) -> pd.DataFrame:
    """全样本逐日RankIC矩阵(date x factor), 与折无关只算一次.
    组内pct-rank后corrwith == spearman(秩相关的定义).
    折内筛选=对本矩阵按train日期切片再取ICIR, 每折独立防selection bias."""
    tr = df[df["tradable"].fillna(False) & df[y_col].notna()]
    cols = factors + [y_col]
    rk = tr.groupby("date")[cols].rank(pct=True)
    rk["date"] = tr["date"].values
    return rk.groupby("date").apply(
        lambda g: g[factors].corrwith(g[y_col]), include_groups=False)


def screen_from_ic(daily_ic: pd.DataFrame, top_k: int = 15) -> list[str]:
    """折内切片 -> |ICIR| top_k. 只用train数据."""
    icir = daily_ic.mean() / daily_ic.std().replace(0, np.nan)
    ok = icir.dropna()
    if len(ok) == 0:
        return list(daily_ic.columns[:top_k])
    return ok.abs().sort_values(ascending=False).head(top_k).index.tolist()


def factor_signs(daily_ic: pd.DataFrame) -> pd.Series:
    """等权合成的方向: train切片IC均值符号."""
    s = np.sign(daily_ic.mean()).replace(0, 1.0)
    return s.fillna(1.0)


def fit_model(train_df: pd.DataFrame, features: list[str], y_col: str,
              val_frac: float = 0.2) -> LGBMRegressor | None:
    """train按时间尾20%做early-stop val, 返回模型对象(SHAP需要)."""
    tr = train_df.dropna(subset=[y_col])
    if len(tr) < 100:
        return None
    dates = np.sort(tr["date"].unique())
    cut = dates[int(len(dates) * (1 - val_frac))] if len(dates) > 20 else None
    if cut is None:
        trn, val = tr, tr
    else:
        trn, val = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        trn[features], trn[y_col],
        eval_X=val[features], eval_y=val[y_col],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                features: list[str], y_col: str,
                val_frac: float = 0.2) -> pd.Series:
    """train按时间尾20%做early-stop val, 预测test. 返回对齐test索引的预测分."""
    model = fit_model(train_df, features, y_col, val_frac)
    if model is None:
        return pd.Series(np.nan, index=test_df.index)
    return pd.Series(model.predict(test_df[features]),
                     index=test_df.index)


def equal_weight_predict(test_df: pd.DataFrame, features: list[str],
                         signs: pd.Series) -> pd.Series:
    """P4等权基线: 因子z分按train内IC符号对齐后等权平均."""
    w = np.array([signs.get(f, 1.0) for f in features])
    return pd.Series(test_df[features].values @ w / len(features),
                     index=test_df.index)
