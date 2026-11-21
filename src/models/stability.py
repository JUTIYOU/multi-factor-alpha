# -*- coding: utf-8 -*-
"""WP4 稳定性监控: PSI / 暴露持续性 / 滚动IC / IC半衰期.

- PSI边界 = 训练窗(首个test起点之前)tradable十分位, 全样本固定(泄漏清单)
- 暴露持续性 = 季末截面rank vs 63交易日前rank的spearman(因子值漂移速度)
- 滚动IC = daily_ic_matrix的63日均值(预测力衰减)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def psi_frame(df: pd.DataFrame, factors: list[str],
              train_end: pd.Timestamp, freq: str = "QE") -> pd.DataFrame:
    """逐季PSI: 边界=train窗十分位固定, expected=0.1/桶.
    返回长表 [factor, quarter, psi, alarm(>0.25)]."""
    d = df[df["tradable"].fillna(False)]
    trd = d[d["date"] <= train_end]
    d = d.assign(quarter=d["date"].dt.to_period("Q"))
    rows = []
    for f in factors:
        ref = trd[f].dropna()
        if len(ref) < 1000:
            continue
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(edges) < 4:
            continue
        ref_counts = np.histogram(ref, edges)[0] / len(ref)
        for q, g in d.groupby("quarter"):
            v = g[f].dropna()
            if len(v) < 500:
                continue
            act = np.histogram(v, edges)[0] / len(v)
            e = np.clip(ref_counts, 1e-4, None)
            a = np.clip(act, 1e-4, None)
            psi = float(((a - e) * np.log(a / e)).sum())
            rows.append({"factor": f, "quarter": str(q), "psi": psi,
                         "alarm": psi > 0.25})
    return pd.DataFrame(rows)


def exposure_persistence(df: pd.DataFrame, factors: list[str],
                         lag: int = 63, freq: str = "QE") -> pd.DataFrame:
    """季末截面rank vs lag日前rank的spearman -> [factor, quarter, persist]."""
    d = df[df["tradable"].fillna(False)]
    dates = np.sort(d["date"].unique())
    qends = pd.Series(dates).groupby(
        pd.Series(dates).dt.to_period("Q")).max()
    pos = {t: i for i, t in enumerate(dates)}
    rows = []
    for f in factors:
        piv = d.pivot_table(index="date", columns="code", values=f)
        rk = piv.rank(axis=1, pct=True).reindex(dates)
        for q, qe in qends.items():
            i = pos[qe]
            if i < lag:
                continue
            a, b = rk.iloc[i], rk.iloc[i - lag]
            v = a.notna() & b.notna()
            if v.sum() < 30:
                continue
            rows.append({"factor": f, "quarter": str(q),
                         "persist": spearmanr(a[v], b[v]).statistic})
    return pd.DataFrame(rows)


def rolling_ic(daily_ic: pd.DataFrame, win: int = 63) -> pd.DataFrame:
    """daily_ic矩阵 -> 63日滚动均值(预测力时变)."""
    return daily_ic.rolling(win, min_periods=win // 2).mean()


def ic_halflife(df: pd.DataFrame, factors: list[str],
                horizons: list[int], ic_fn) -> pd.DataFrame:
    """IC(h)衰减曲线+半衰期(IC<=0.5*IC(1)的最小h).
    ic_fn: (df, factors, ret_col) -> daily_ic矩阵."""
    rows = []
    for h in horizons:
        m = ic_fn(df, factors, f"fwd_ret_{h}").mean()
        for f, v in m.items():
            rows.append({"factor": f, "h": h, "ic": v})
    long = pd.DataFrame(rows)
    half = {}
    for f, g in long.groupby("factor"):
        g = g.set_index("h").sort_index()["ic"]
        base = g.iloc[0]
        h2 = g[abs(g) <= abs(base) / 2]
        half[f] = int(h2.index[0]) if len(h2) else f">{int(g.index[-1])}"
    return long.assign(
        halflife=long["factor"].map(half))


def drift_summary(psi: pd.DataFrame, persist: pd.DataFrame,
                  ric: pd.DataFrame) -> pd.DataFrame:
    """汇总: 每因子 maxPSI/报警季数/平均持续性/IC前后段比."""
    p = (psi.groupby("factor")
         .agg(psi_max=("psi", "max"), psi_mean=("psi", "mean"),
              alarm_q=("alarm", "sum")))
    pe = persist.groupby("factor")["persist"].mean().rename("persist_mean")
    n = len(ric)
    early = ric.iloc[: n // 3].mean().rename("ic_early")
    late = ric.iloc[-n // 3:].mean().rename("ic_late")
    out = pd.concat([p, pe, early, late], axis=1)
    out["ic_decay_ratio"] = out["ic_late"] / out["ic_early"].replace(
        0, np.nan)
    out.index.name = "factor"
    return out.reset_index()
