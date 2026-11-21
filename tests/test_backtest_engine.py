# -*- coding: utf-8 -*-
"""回测引擎合成测试: 固定权重记账/边界日相加/续留免费.

防泄漏清单#20回归测试: 组合日收益必须=Σr_i/N(固定权重),
不能用"当日在场腿均值"——顺延腿只剩零星股票时均值会把个股日行
放大成全组合收益(该bug曾7年虚增+157%).
"""
import numpy as np
import pandas as pd

from src.backtest.engine import daily_group_returns, nav_with_costs

CAL = pd.date_range("2024-01-01", periods=4, freq="B").values  # d0..d3
D = pd.date_range("2024-01-01", periods=4, freq="B")


def _legs(rows):
    return pd.DataFrame(
        rows, columns=["reb", "group", "code", "entry", "exit", "ret"])


def _px(rows):
    return pd.DataFrame(
        rows, columns=["code", "date", "open_badj", "close_badj",
                       "preclose_badj"])


def test_fixed_weight_not_present_mean():
    """顺延腿只剩1只在场时, 当日组合收益=r/3而非r."""
    # reb=d0信号, a持有到d3(顺延), b/c到期d2退出; d3只剩a在场且跌10%
    legs = _legs([
        (D[0], 5, "a", D[1], D[3], -0.06),
        (D[0], 5, "b", D[1], D[2], 0.01),
        (D[0], 5, "c", D[1], D[2], 0.01),
    ])
    px = _px([
        ("a", D[1], 1.0, 1.0, 1.0),          # entry日: close/open-1=0
        ("a", D[2], 1.0, 1.0, 1.0),          # 中间日: 0
        ("a", D[3], 0.9, 0.9, 1.0),          # exit日: open/preclose-1=-0.1
        ("b", D[1], 1.0, 1.0, 1.0),
        ("b", D[2], 1.01, 1.01, 1.0),        # exit: +0.01
        ("c", D[1], 1.0, 1.0, 1.0),
        ("c", D[2], 1.01, 1.01, 1.0),
    ])
    out = daily_group_returns(legs, px, CAL)
    r3 = out[out["date"] == D[3]]["ret"].iloc[0]
    # 正确: -0.1/3≈-0.033; 旧bug(在场均值): -0.1
    assert np.isclose(r3, -0.1 / 3)
    assert not np.isclose(r3, -0.1)
    # d2: b,c各+0.01/3, a为0 -> +0.01*2/3
    r2 = out[out["date"] == D[2]]["ret"].iloc[0]
    assert np.isclose(r2, 0.01 * 2 / 3)


def test_boundary_day_adds_and_costs():
    """边界日=旧exit+新entry相加; 换仓100%时双边成本记在该日."""
    # reb1信号d0->a d1入d2出; reb2信号d1->x d2入d3出; d2=边界日
    legs = _legs([
        (D[0], 5, "a", D[1], D[2], 0.05),   # a: d1入d2出
        (D[1], 5, "x", D[2], D[3], 0.02),   # x: d2入d3出, d2即边界日
    ])
    px = _px([
        ("a", D[1], 1.0, 1.0, 1.0),
        ("a", D[2], 1.05, 1.05, 1.0),       # exit: +0.05
        ("x", D[2], 1.0, 1.02, 1.0),        # entry: close/open-1=+0.02
        ("x", D[3], 1.02, 1.02, 1.02),      # exit: 0
    ])
    daily = daily_group_returns(legs, px, CAL)
    nav = nav_with_costs(daily, legs, buy_bp=5, sell_bp=15,
                         group=5).set_index("date")
    # 边界日d2: 旧exit +0.05(N=1) + 新entry +0.02(N=1) = 0.07
    assert np.isclose(nav.loc[D[2], "ret_gross"], 0.07)
    # 成本: reb1建仓只买(5bp); d2卖旧(15bp)+买新(5bp)=20bp
    assert np.isclose(nav.loc[D[1], "ret_net"], -5 / 1e4)
    assert np.isclose(nav.loc[D[2], "ret_net"], 0.07 - 20 / 1e4)


def test_continuing_holding_no_cost():
    """名单完全不变(keep=100%): 边界日不扣双边成本."""
    legs = _legs([
        (D[0], 5, "a", D[1], D[2], 0.05),
        (D[1], 5, "a", D[2], D[3], 0.02),   # 同一code续留
    ])
    px = _px([
        ("a", D[1], 1.0, 1.0, 1.0),
        ("a", D[2], 1.0, 1.02, 1.0),        # 旧exit +0 与 新entry +0.02
        ("a", D[3], 1.02, 1.02, 1.02),
    ])
    daily = daily_group_returns(legs, px, CAL)
    nav = nav_with_costs(daily, legs, buy_bp=5, sell_bp=15,
                         group=5).set_index("date")
    # keep=100% -> sell_f=buy_f=0 -> d2只记收益不扣成本
    assert np.isclose(nav.loc[D[2], "ret_gross"], 0.02)
    assert np.isclose(nav.loc[D[2], "ret_net"], 0.02)
