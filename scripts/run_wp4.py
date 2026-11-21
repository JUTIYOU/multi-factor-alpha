# -*- coding: utf-8 -*-
"""WP4 编排: SHAP归因审计 + PSI/CSI稳定性 + IC半衰期.

产出:
- reports/fold_selection.csv   27折x2周期 每折入选因子+ICIR
- reports/shap_importance.csv  因子级+簇级重要性、折间排名稳定性
- reports/factor_drift.csv     PSI/暴露持续性/早晚段IC比
- reports/ic_halflife.csv      h=1..20 IC衰减+半衰期
- reports/composite_drift.csv  P3/P4合成信号滚动IC统计
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import labels, splits, train as tr
from src.models import shap_audit as sa, stability as st
from src.factors import ic as icmod
from src.data.baostock_client import load_yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TEST_LEN, MIN_TRAIN, TOP_K = 63, 504, 15


def main() -> None:
    cfg = load_yaml(ROOT / "configs" / "factors.yaml")
    factors = [k for sec in ("price_volume", "valuation", "fundamental")
               for k, v in cfg[sec].items() if v.get("neutralize")]
    print("[wp4] load indsize frame", flush=True)
    fac = pd.read_parquet(PROC / "factors_indsize.parquet")
    pan = pd.read_parquet(PROC / "panel.parquet",
                          columns=["code", "date", "tradable", "close_badj"])
    df = pan.merge(fac, on=["code", "date"], how="left")
    df = icmod.add_fwd_ret(df, list(range(1, 21)))
    df = labels.add_labels(df, [5, 10])
    use_f = [f for f in factors if f in df.columns]
    dates = df["date"]
    train_end = np.sort(dates.unique())[MIN_TRAIN - 1]
    train_df = df[dates <= train_end]
    print(f"[wp4] train_end={pd.Timestamp(train_end).date()}", flush=True)

    # ---- 1. 折间筛选名单重建(纯函数,无需重训) ----
    print("[wp4] rebuild fold selections", flush=True)
    clusters = sa.factor_clusters(train_df, use_f)
    sel_rows, fold_feats, shap_long, dailycs = [], {}, [], {}
    for h in (5, 10):
        t0 = time.time()
        dailyc = tr.daily_ic_matrix(df, use_f, f"y_{h}")
        dailycs[h] = dailyc
        folds = splits.wf_folds(dates, TEST_LEN, MIN_TRAIN,
                                purge_h=h, embargo=h)
        for i, (tri, _) in enumerate(folds):
            tmax = dates.iloc[tri].max()
            sl = dailyc.loc[:tmax]
            feats = tr.screen_from_ic(sl, TOP_K)
            fold_feats[(h, i)] = feats
            icir = (sl.mean() / sl.std().replace(0, np.nan))
            for r, f in enumerate(feats):
                sel_rows.append({"horizon": h, "fold": i, "factor": f,
                                 "rank": r + 1, "icir": icir.get(f)})
        print(f"[wp4] h={h} selections ({time.time() - t0:.0f}s)",
              flush=True)
        # ---- 2. SHAP审计 ----
        long = sa.run_shap_audit(df, folds, fold_feats, clusters,
                                 f"y_{h}", h)
        shap_long.append(long)
        stab = sa.rank_stability(
            long.pivot_table(index="fold", columns="factor",
                             values="rank").dropna(axis=1, how="any"))
        print(f"[wp4] h={h} shap done, fold-rank corr={stab:.3f}",
              flush=True)
    pd.DataFrame(sel_rows).to_csv(REPORTS / "fold_selection.csv",
                                  index=False)
    shap_sum = sa.summarize_shap(pd.concat(shap_long))
    shap_sum.to_csv(REPORTS / "shap_importance.csv", index=False)
    print("[wp4] -> fold_selection.csv, shap_importance.csv", flush=True)

    # ---- 2b. 复现性验证: 重训fold0预测 vs wp3_preds ----
    pr = pd.read_parquet(PROC / "wp3_preds.parquet")
    pr = pr[(pr.protocol == "P3") & (pr.horizon == 5)]
    folds5 = splits.wf_folds(dates, TEST_LEN, MIN_TRAIN,
                             purge_h=5, embargo=5)
    tri, tei = folds5[0]
    m = tr.fit_model(df.iloc[tri], fold_feats[(5, 0)], "y_5")
    rep = pd.Series(m.predict(df.iloc[tei][fold_feats[(5, 0)]]),
                    index=df.iloc[tei].index)
    chk = df.iloc[tei][["code", "date"]].assign(pred=rep).merge(
        pr[["code", "date", "pred"]], on=["code", "date"],
        suffixes=("_new", "_old"))
    c = chk["pred_new"].corr(chk["pred_old"])
    print(f"[wp4] retrain-vs-saved fold0 corr={c:.6f}", flush=True)

    # ---- 3. 稳定性 ----
    print("[wp4] PSI", flush=True)
    psi = st.psi_frame(df, use_f, train_end)
    print("[wp4] persistence", flush=True)
    persist = st.exposure_persistence(df, use_f)
    ric = st.rolling_ic(dailycs[5])
    drift = st.drift_summary(psi, persist, ric)
    drift.to_csv(REPORTS / "factor_drift.csv", index=False)
    print("[wp4] -> factor_drift.csv", flush=True)

    # ---- 4. IC半衰期 ----
    print("[wp4] halflife h=1..20", flush=True)
    hl = st.ic_halflife(df, use_f, list(range(1, 21)),
                        tr.daily_ic_matrix)
    hl.to_csv(REPORTS / "ic_halflife.csv", index=False)
    print("[wp4] -> ic_halflife.csv", flush=True)

    # ---- 5. 合成信号漂移 ----
    rows = []
    allp = pd.read_parquet(PROC / "wp3_preds.parquet")
    for (h, proto), g in allp.groupby(["horizon", "protocol"]):
        rc = f"fwd_ret_{h}"
        d = g[g["tradable"].fillna(False)
              & g.pred.notna() & g[rc].notna()]
        icd = d.groupby("date").apply(
            lambda x: x.pred.corr(x[rc], method="spearman")
            if len(x) > 20 else np.nan, include_groups=False).dropna()
        r = icd.rolling(63, min_periods=30).mean()
        rows.append({"horizon": h, "protocol": proto,
                     "ic_mean": icd.mean(),
                     "roll_min": r.min(), "roll_max": r.max(),
                     "roll_last": r.iloc[-1],
                     "neg_days_pct": (icd < 0).mean()})
    pd.DataFrame(rows).to_csv(REPORTS / "composite_drift.csv",
                              index=False)
    print("[wp4] -> composite_drift.csv; ALL DONE", flush=True)


if __name__ == "__main__":
    main()
