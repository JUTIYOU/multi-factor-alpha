# -*- coding: utf-8 -*-
"""从 reports/*.csv 绘制面试/报告用图 -> figures/*.png.

只读产物文件, 不依赖 panel(241万行); 基准曲线读 data/raw/index_quotes.parquet.
运行: .venv/Scripts/python scripts/plot_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP, FIG, RAW = ROOT / "reports", ROOT / "figures", ROOT / "data" / "raw"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def fig1_nav():
    """净值曲线(主线P4 h10): net/gross + 中证500 + 回撤."""
    nav = pd.read_csv(REP / "backtest_nav.csv", parse_dates=["date"])
    d = nav[(nav.protocol == "P4") & (nav.horizon == 10)]
    idx = pd.read_parquet(RAW / "index_quotes.parquet")
    idx["close"] = pd.to_numeric(idx["close"])
    idx["date"] = pd.to_datetime(idx["date"])
    bench = idx.set_index("date")["close"].reindex(d["date"]).ffill()
    bench_nav = bench / bench.iloc[0]
    dd = d["nav_net"] / d["nav_net"].cummax() - 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(d["date"], d["nav_net"], label="P4 h10 net (after cost)",
             lw=1.4, color="tab:blue")
    ax1.plot(d["date"], d["nav_gross"], label="gross (no cost)",
             lw=1, ls="--", color="tab:blue", alpha=0.6)
    ax1.plot(d["date"], bench_nav, label="CSI 500 (benchmark)",
             lw=1, color="gray")
    ax1.set_ylabel("NAV")
    ax1.set_title("Long-only top-quintile portfolio vs CSI 500 "
                  "(2019-02 to 2025-12)")
    ax1.legend()
    ax2.fill_between(d["date"], dd * 100, 0, color="tab:red", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_nav.png")
    plt.close(fig)


def fig2_ic_decay():
    """IC衰减曲线: 代表性因子 h=1..20."""
    ic = pd.read_csv(REP / "ic_halflife.csv")
    picks = ["rev_5", "mom_12_1", "ivol", "turn_mean", "roe", "lev", "BP"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for f in picks:
        s = ic[ic.factor == f].sort_values("h")
        if len(s):
            ax.plot(s["h"], s["ic"], marker="o", ms=3, lw=1.2, label=f)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("holding horizon h (trading days)")
    ax.set_ylabel("daily mean RankIC")
    ax.set_title("IC decay by horizon — slow (liquidity/vol) vs fast "
                 "(reversal) signals")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ic_decay.png")
    plt.close(fig)


def fig3_neutralization():
    """raw vs indsize: IC/ICIR散点(y=x参考线) -> 风格伪装占比."""
    n = pd.read_csv(REP / "factor_neutralization.csv")
    n = n[n.horizon == 5]
    raw = n[n.variant == "raw"].set_index("factor")
    ind = n[n.variant == "indsize"].set_index("factor")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, col, ttl in [(axes[0], "ic_mean", "mean daily IC"),
                         (axes[1], "icir", "ICIR")]:
        x, y = raw[col], ind[col].reindex(raw.index)
        ax.scatter(x.abs(), y.abs(), s=28, alpha=0.75)
        lim = max(x.abs().max(), y.abs().max()) * 1.08
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="no decay")
        for f in ["mom_12_1", "turn_mean", "roe", "lev"]:
            if f in x.index:
                ax.annotate(f, (abs(x[f]), abs(y[f])), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(f"raw |{col}|")
        ax.set_ylabel(f"indsize |{col}|")
        ax.set_title(f"{ttl}: raw vs industry+size neutralized")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Neutralization cost: points below the line lost "
                 "predictive power to style removal", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_neutralization.png", bbox_inches="tight")
    plt.close(fig)


def fig4_shap():
    """SHAP簇级份额: h5 vs h10 分组条形."""
    s = pd.read_csv(REP / "shap_importance.csv")
    cl = (s.drop_duplicates(["horizon", "cluster"])
            .pivot(index="cluster", columns="horizon",
                   values="cluster_share")
            .fillna(0))
    cl = cl.sort_values(cl.columns[-1], ascending=True)
    cl.index.name = None
    (cl * 100).plot.barh(figsize=(7.5, 4.5))
    plt.xlabel("share of total |SHAP| (%)")
    plt.title("Cluster-level SHAP importance (P3 model, OOS folds)")
    plt.legend(title="horizon", labels=[f"h={c}" for c in cl.columns])
    plt.tight_layout()
    plt.savefig(FIG / "fig4_shap.png")
    plt.close()


def fig5_attribution():
    """归因: P4 h10累计成分堆叠面积 + 4臂总分解柱状."""
    a = pd.read_csv(REP / "backtest_attribution.csv",
                    parse_dates=["reb"])
    d = a[(a.protocol == "P4") & (a.horizon == 10)].sort_values("reb")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    comp = ["cum_market", "cum_industry", "cum_size", "cum_residual"]
    labels = ["market", "industry", "size", "residual (alpha)"]
    ax.stackplot(d["reb"], *[d[c] for c in comp], labels=labels,
                 alpha=0.8)
    ax.plot(d["reb"], d["cum_ret"], color="k", lw=1.2,
            label="total")
    ax.set_title("Cumulative return decomposition — P4 h10")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("cumulative return")
    ax = axes[1]
    last = (a.sort_values("reb").groupby(["protocol", "horizon"])
             [["cum_market", "cum_industry", "cum_size",
               "cum_residual"]].last())
    last.index = [f"{p} h{h}" for p, h in last.index]
    (last * 100).plot.bar(stacked=True, ax=ax, rot=0)
    ax.set_ylabel("cumulative contribution (%)")
    ax.set_title("Final additive components by arm")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_attribution.png")
    plt.close(fig)


def fig6_drift():
    """漂移归因散点: psi_max vs ic_decay_ratio 四象限."""
    d = pd.read_csv(REP / "factor_drift.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["psi_max"], d["ic_decay_ratio"], s=45, alpha=0.7)
    ax.axvline(0.25, color="tab:red", ls="--", lw=1,
               label="PSI alarm = 0.25")
    ax.axhline(1.0, color="gray", ls="--", lw=1,
               label="late/early IC = 1")
    for _, r in d.iterrows():
        if r["psi_max"] > 0.4 or abs(r["ic_decay_ratio"]) < 0.5 \
                or r["factor"] in ("ivol", "turn_mean"):
            ax.annotate(r["factor"], (r["psi_max"], r["ic_decay_ratio"]),
                        fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("max PSI (distribution drift)")
    ax.set_ylabel("IC late/early ratio (predictive decay)")
    ax.set_title("Drift vs decay are different axes — high-PSI factors "
                 "are not necessarily dead")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig6_drift.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1_nav, fig2_ic_decay, fig3_neutralization,
               fig4_shap, fig5_attribution, fig6_drift):
        fn()
        print("done:", fn.__name__)
    print("-> figures/")
