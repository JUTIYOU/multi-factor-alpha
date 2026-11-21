# -*- coding: utf-8 -*-
"""PIT机制单测: 不依赖已拉数据,用合成数据断言机制正确性.

运行: .venv/Scripts/python -m pytest tests/test_pit.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pit import attach_available_date, asof_join, pit_check, IMPUTE_DAYS


def make_fin():
    """code A 两期财报: 2024Q1(4-28公告) / 2024Q2(8-30公告);
    code B 一期pubDate缺失."""
    return pd.DataFrame({
        "code": ["A", "A", "B"],
        "statDate": ["2024-03-31", "2024-06-30", "2024-03-31"],
        "pubDate": ["2024-04-28", "2024-08-30", None],
        "roeAvg": [3.0, 7.0, 2.5],
    })


def test_attach_available_date():
    fin = attach_available_date(make_fin())
    a = fin[fin["code"] == "A"].sort_values("statDate")
    assert a["available_date"].tolist() == [pd.Timestamp("2024-04-28"),
                                            pd.Timestamp("2024-08-30")]
    b = fin[fin["code"] == "B"].iloc[0]
    assert b["available_date"] == pd.Timestamp("2024-03-31") + pd.Timedelta(days=IMPUTE_DAYS)
    assert bool(b["imputed"]) is True


def test_asof_join_never_sees_future():
    fin = attach_available_date(make_fin())
    panel = pd.DataFrame({
        "code": ["A"] * 4 + ["B", "C"],
        "date": pd.to_datetime(["2024-04-27", "2024-04-28", "2024-08-29",
                                 "2024-08-30", "2024-12-31", "2024-06-01"]),
    })
    m = asof_join(panel, fin, fields=["roeAvg"], suffix="fin_")
    m = m.sort_values(["code", "date"]).reset_index(drop=True)
    a = m[m["code"] == "A"].reset_index(drop=True)
    # 公告前一天 -> 无可用记录
    assert pd.isna(a.loc[0, "fin_roeAvg"])
    # 公告当天 -> 可见Q1
    assert a.loc[1, "fin_roeAvg"] == 3.0
    # 中报公告前一天 -> 仍是Q1
    assert a.loc[2, "fin_roeAvg"] == 3.0
    # 中报公告当天 -> 切到Q2
    assert a.loc[3, "fin_roeAvg"] == 7.0
    # B: pubDate缺失 -> 兜底日(6-29)后可见
    assert m[m["code"] == "B"]["fin_roeAvg"].iloc[0] == 2.5
    # C: 无财报 -> 全NaN
    assert m[m["code"] == "C"]["fin_roeAvg"].isna().all()
    assert pit_check(m)


def test_asof_join_multirow_same_day():
    """同一code同日多条panel记录(去重场景)也保持一致."""
    fin = attach_available_date(make_fin())
    panel = pd.DataFrame({
        "code": ["A", "A"],
        "date": pd.to_datetime(["2024-05-10", "2024-05-10"]),
    })
    m = asof_join(panel, fin, fields=["roeAvg"], suffix="fin_")
    assert (m["fin_roeAvg"] == 3.0).all()


def test_future_report_invisible():
    """关键断言: 公告日前的panel行绝不能join到该期财报."""
    fin = attach_available_date(make_fin())
    panel = pd.DataFrame({
        "code": ["A"],
        "date": pd.to_datetime(["2024-04-27"]),  # Q1公告前一天
    })
    m = asof_join(panel, fin, fields=["roeAvg"], suffix="fin_")
    assert m["fin_roeAvg"].isna().all()
    assert m["_fin_available_date"].isna().all()
