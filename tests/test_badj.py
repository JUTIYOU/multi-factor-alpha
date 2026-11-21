# -*- coding: utf-8 -*-
"""后复权回归测试(泄漏清单#19): backAdjustFactor是累计因子,
直接as-of取值, 禁止cumprod.

合成: 两次除权事件, BF=1.10(d2)和1.20(d3)为累计值.
若错误cumprod, d3会得到1.10*1.20=1.32; 正确应为1.20.
"""
import numpy as np
import pandas as pd

from src.data.panel import add_badj

D = pd.date_range("2024-01-01", periods=4, freq="B")


def _panel():
    # 单票4日: close恒10; d2除权(f_true=1.1), d3除权(f_true=1.2)
    return pd.DataFrame({
        "code": "a", "date": D,
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "preclose": [10.0, 10.0, 10 / 1.1, 10 / 1.2],
        "tradestatus": 1, "pctChg": 0.0, "isST": "0",
    })


def test_cumulative_factor_not_cumproded():
    adj = pd.DataFrame({
        "code": ["a", "a"],
        "dividOperateDate": [D[2], D[3]],
        "backAdjustFactor": [1.10, 1.20],  # 累计因子
    })
    out = add_badj(_panel(), adj).set_index("date")
    # 首事件前: 基线=BF(首事件)/f_true = 1.10/1.10 = 1.0
    assert np.isclose(out.loc[D[0], "badj_cum"], 1.0)
    assert np.isclose(out.loc[D[1], "badj_cum"], 1.0)
    # 事件日及之后: as-of取累计值, 不是累乘
    assert np.isclose(out.loc[D[2], "badj_cum"], 1.10)
    assert np.isclose(out.loc[D[3], "badj_cum"], 1.20)
    # 若cumprod则这里会是1.32 —— 显式否定
    assert not np.isclose(out.loc[D[3], "badj_cum"], 1.32)
    assert np.isclose(out.loc[D[3], "close_badj"], 12.0)


def test_no_events_defaults_one():
    adj = pd.DataFrame(columns=["code", "dividOperateDate",
                                "backAdjustFactor"])
    out = add_badj(_panel(), adj).set_index("date")
    assert (out["badj_cum"] == 1.0).all()
    assert np.isclose(out.loc[D[3], "close_badj"], 10.0)
