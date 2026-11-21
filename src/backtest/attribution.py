# -*- coding: utf-8 -*-
"""组合风险归因: 市场/行业/风格size/残差 四段分解.

每期对全体可交易腿截面回归:
  ret_i = 行业哑变量(无截距) + beta * size_i(中心化)
  ind_coef[ind] = 行业绝对收益; 行业超额 = ind_coef - 全样本均值
组合(等权)分解: ret = market(全样本均值) + w·ind_excess + w·size + w·resid
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attribute_period(legs_univ: pd.DataFrame, port_codes: list,
                     ind_map: pd.Series, size_map: pd.Series) -> dict:
    """单期分解. legs_univ: 当期全部有效腿(code,ret).
    ind_map/size_map: code -> industry / ln_mktcap(信号日口径)."""
    d = legs_univ[legs_univ["ret"].notna()].copy()
    d["ind"] = d["code"].map(ind_map).fillna("UNKNOWN")
    d["size"] = d["code"].map(size_map)
    d = d[d["size"].notna()]
    if len(d) < 50:
        return {}
    D = pd.get_dummies(d["ind"]).astype(float)
    size_c = d["size"] - d["size"].mean()
    X = np.column_stack([D.values, size_c.values])
    coef, *_ = np.linalg.lstsq(X, d["ret"].values, rcond=None)
    ind_coef = pd.Series(coef[:D.shape[1]], index=D.columns)
    beta = coef[-1]
    d["fit_ind"] = d["ind"].map(ind_coef)
    d["fit_size"] = beta * (d["size"] - d["size"].mean())
    d["resid"] = d["ret"] - d["fit_ind"] - d["fit_size"]
    mkt = d["ret"].mean()
    port = d[d["code"].isin(port_codes)]
    if len(port) == 0:
        return {}
    w = 1.0 / len(port)
    return {
        "n": len(port),
        "ret": port["ret"].mean(),
        "market": mkt,
        # 行业哑变量系数含市场水平 -> 减全样本均值=行业超额贡献
        "industry": w * (port["fit_ind"] - mkt).sum(),
        "size": w * port["fit_size"].sum(),
        "residual": w * port["resid"].sum(),
    }


def attribute_all(legs: pd.DataFrame, ind_by_reb: dict,
                  size_by_reb: dict, group: int = 5) -> pd.DataFrame:
    """逐期归因 -> 长表 [reb, market, industry, size, residual, ret].
    ind_by_reb/size_by_reb: 信号日 -> code级Series映射(PIT口径)."""
    rows = []
    for reb, g in legs.groupby("reb"):
        ports = g[(g["group"] == group) & g["ret"].notna()]["code"].tolist()
        r = attribute_period(g, ports, ind_by_reb.get(reb, {}),
                             size_by_reb.get(reb, {}))
        if r:
            r["reb"] = reb
            rows.append(r)
    out = pd.DataFrame(rows)
    for c in ("market", "industry", "size", "residual", "ret"):
        out[f"cum_{c}"] = (1 + out[c]).cumprod() - 1
    return out
