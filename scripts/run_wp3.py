# -*- coding: utf-8 -*-
"""WP3 编排: 嵌套筛选 + 4协议对比.

协议: P1单折不重训 / P2走样不切边 / P3走样+purge(h)+embargo(h) [主协议] / P4等权基线(P3协议)
矩阵: {P1,P2,P3,P4} × h{5,10} × indsize特征; 另 P3×raw特征 消融.
输出: reports/model_protocol_comparison.csv + data/processed/wp3_preds.parquet(P3/P4 OOS预测)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import labels, splits, train as tr, eval as ev
from src.factors import ic as icmod
from src.data.baostock_client import load_yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HORIZONS = [5, 10]
TEST_LEN, MIN_TRAIN, TOP_K = 63, 504, 15


def load_frame(variant: str) -> pd.DataFrame:
    fac = pd.read_parquet(PROC / f"factors_{variant}.parquet")
    pan = pd.read_parquet(PROC / "panel.parquet",
                          columns=["code", "date", "tradable", "close_badj"])
    df = pan.merge(fac, on=["code", "date"], how="left")
    df = icmod.add_fwd_ret(df, HORIZONS)
    return labels.add_labels(df, HORIZONS)


def fold_screen(dailyc: pd.DataFrame, tr_idx: np.ndarray,
                dates: pd.Series, top_k: int = TOP_K):
    """折内IC切片(dates<=train末日) -> (入选因子, 符号)."""
    tmax = dates.iloc[tr_idx].max()
    sl = dailyc.loc[:tmax]
    feats = tr.screen_from_ic(sl, top_k)
    return feats, tr.factor_signs(sl)


def run_arm(df: pd.DataFrame, factors: list[str], y_col: str,
            protocol: str, purge_h: int,
            dailyc: pd.DataFrame) -> pd.Series:
    """按协议产出全测试区OOS预测序列(对齐df索引)."""
    pred = pd.Series(np.nan, index=df.index)
    if protocol == "P1":
        tr_idx, te_idx = splits.p1_split(df["date"], MIN_TRAIN)[0]
        feats, _ = fold_screen(dailyc, tr_idx, df["date"])
        p = tr.fit_predict(df.iloc[tr_idx], df.iloc[te_idx],
                           feats, y_col)
        pred.loc[p.index] = p
        return pred
    folds = splits.wf_folds(df["date"], TEST_LEN, MIN_TRAIN,
                            purge_h=purge_h, embargo=purge_h)
    for i, (tr_idx, te_idx) in enumerate(folds):
        trdf, tedf = df.iloc[tr_idx], df.iloc[te_idx]
        feats, signs = fold_screen(dailyc, tr_idx, df["date"])
        if protocol == "P4":
            p = tr.equal_weight_predict(tedf, feats, signs)
        else:
            p = tr.fit_predict(trdf, tedf, feats, y_col)
        pred.loc[p.index] = p
        print(f"    fold {i + 1}/{len(folds)} n_feat={len(feats)}",
              flush=True)
    return pred


def main() -> None:
    if not (PROC / "factors_indsize.parquet").exists():
        sys.exit("先跑 scripts/build_variants.py")
    cfg = load_yaml(ROOT / "configs" / "factors.yaml")
    factors = [k for sec in ("price_volume", "valuation", "fundamental")
               for k, v in cfg[sec].items() if v.get("neutralize")]
    results, preds_out = [], []
    for variant, protos in [("indsize", ["P1", "P2", "P3", "P4"]),
                            ("raw", ["P3"])]:
        print(f"[wp3] load {variant}", flush=True)
        df = load_frame(variant)
        use_f = [f for f in factors if f in df.columns]
        for h in HORIZONS:
            y_col, ret_col = f"y_{h}", f"fwd_ret_{h}"
            t0 = time.time()
            dailyc = tr.daily_ic_matrix(df, use_f, y_col)
            print(f"[wp3] daily_ic_matrix {variant} h={h} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            for proto in protos:
                t0 = time.time()
                purge = h if proto in ("P3", "P4") else 0
                print(f"[wp3] {variant} h={h} {proto} purge={purge}",
                      flush=True)
                pred = run_arm(df, use_f, y_col, proto, purge, dailyc)
                df["_pred"] = pred
                r = ev.eval_signal(df, "_pred", ret_col)
                r.update(variant=variant, horizon=h, protocol=proto,
                         secs=round(time.time() - t0, 1))
                results.append(r)
                if variant == "indsize" and proto in ("P3", "P4"):
                    preds_out.append(df[["code", "date", "tradable",
                                         ret_col]].assign(
                        pred=pred, horizon=h, protocol=proto))
                print(f"    -> IC={r['ic_mean']:.4f} ICIR={r['icir']:.3f} "
                      f"({r['secs']}s)", flush=True)
        del df
    out = pd.DataFrame(results)[
        ["variant", "horizon", "protocol", "ic_mean", "ic_std", "icir",
         "ic_pos", "top_bottom", "spread_std", "n_days", "secs"]]
    out.to_csv(REPORTS / "model_protocol_comparison.csv", index=False)
    print("[wp3] -> reports/model_protocol_comparison.csv", flush=True)
    if preds_out:
        pd.concat(preds_out).to_parquet(PROC / "wp3_preds.parquet",
                                      index=False)
        print("[wp3] -> wp3_preds.parquet", flush=True)


if __name__ == "__main__":
    main()
