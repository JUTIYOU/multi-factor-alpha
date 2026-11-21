# -*- coding: utf-8 -*-
"""WP3合成测试: splits边界/purge语义 + 端到端小样本跑通."""
import numpy as np
import pandas as pd

from src.models import labels, splits, train as tr, eval as ev


def _frame(n_days=40, n_codes=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    df = pd.DataFrame(
        [(c, d) for d in dates for c in range(n_codes)],
        columns=["code", "date"])
    df["tradable"] = True
    df["f1"] = rng.standard_normal(len(df))
    df["f2"] = rng.standard_normal(len(df))
    df["fwd_ret_5"] = df["f1"] * 0.5 + rng.standard_normal(len(df)) * 0.1
    df["fwd_ret_10"] = df["f1"] * 0.5
    return df


def test_fold_last_keeps_tail():
    d = pd.date_range("2020-01-01", periods=100, freq="B")
    ds = pd.Series(np.repeat(d, 3))
    folds = splits.wf_folds(ds, test_len=10, min_train=50)
    covered = np.concatenate([te for _, te in folds])
    # 所有>=第50天的日期都必须被某折覆盖(尾部不丢)
    assert covered.max() == len(ds) - 1
    assert ds.iloc[covered].min() >= d[50]


def test_purge_embargo_gap():
    d = pd.date_range("2020-01-01", periods=120, freq="B")
    ds = pd.Series(np.repeat(d, 2))
    tr, te = splits.wf_folds(ds, test_len=10, min_train=60,
                             purge_h=5, embargo=5)[0]
    gap = (d[60] - ds.iloc[tr].max()).days
    # train最末日期与test首日之间至少purge+embargo=10个工作日跨度
    assert ds.iloc[tr].max() <= d[60 - 10]


def test_p1_train_once():
    d = pd.date_range("2020-01-01", periods=100, freq="B")
    ds = pd.Series(np.repeat(d, 4))
    tr_idx, te_idx = splits.p1_split(ds, 50)[0]
    assert ds.iloc[tr_idx].max() < ds.iloc[te_idx].min()
    assert len(te_idx) == 50 * 4


def test_end_to_end_signal_found():
    df = _frame()
    df = labels.add_labels(df, [5])
    trdf, tedf = df[df["date"] < "2020-02-01"], df[df["date"] >= "2020-02-01"]
    dailyc = tr.daily_ic_matrix(trdf, ["f1", "f2"], "y_5")
    feats = tr.screen_from_ic(dailyc, top_k=1)
    assert feats == ["f1"]  # f1是真信号,必被选中
    p = tr.fit_predict(trdf, tedf, feats, "y_5")
    tedf = tedf.assign(pred=p)
    r = ev.eval_signal(tedf, "pred", "fwd_ret_5")
    assert r["ic_mean"] > 0.05  # 合成信号必须显著正
