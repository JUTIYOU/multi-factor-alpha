# -*- coding: utf-8 -*-
"""WP3 训练协议切分: P1单折 / P2走样不切边 / P3走样+purge+embargo.

- test折: 每63交易日一折, 首折起点保证train>=min_train(默认504日≈2年)
- P2: train=[起点, T), test=[T, T+63) 扩展窗
- P3: train=[起点, T-h-embargo), 砍掉标签窗伸进test的尾部样本(purge=h),
      再留embargo间隔
- P1: 只用首个test折起点之前的全部数据训一次, 在所有test折上评估
返回 (train_mask, test_mask) 为日期的boolean索引序列(对dates数组).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wf_folds(dates: pd.Series | np.ndarray, test_len: int = 63,
             min_train: int = 504, purge_h: int = 0,
             embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """walk-forward折. purge/embargo按h传参(P2传0,P3传h).
    返回 [(train_idx, test_idx)] 位置索引."""
    d = np.asarray(pd.to_datetime(dates))
    uniq = np.unique(d)
    folds = []
    t = min_train
    while t < len(uniq):
        cut = uniq[t]
        # purge: train中t>=T-h的样本其标签窗(t,t+h]与test重叠 -> 必须砍;
        # embargo再额外留间隔. tr_end是train允许的最后一个unique索引
        tr_end = t - purge_h - embargo
        tr = np.where(d < uniq[tr_end])[0] if tr_end > 0 else np.array([], int)
        if t + test_len < len(uniq):
            te = np.where((d >= cut) & (d < uniq[t + test_len]))[0]
        else:
            te = np.where(d >= cut)[0]
        if len(tr) and len(te):
            folds.append((tr, te))
        t += test_len
    return folds


def p1_split(dates: pd.Series | np.ndarray, test_start_idx: int,
             test_len: int = 63) -> list[tuple[np.ndarray, np.ndarray]]:
    """单折: train=min_train之前的窗到首个test起点, test=全部后续折并集."""
    d = np.asarray(pd.to_datetime(dates))
    uniq = np.unique(d)
    cut = uniq[test_start_idx]
    tr = np.where(d < cut)[0]
    te = np.where(d >= cut)[0]
    return [(tr, te)]


def first_test_idx(dates: pd.Series | np.ndarray,
                   min_train: int = 504) -> int:
    return min_train
