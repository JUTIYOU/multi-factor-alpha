# -*- coding: utf-8 -*-
"""WP2 合成数据冒烟: 不依赖真实panel, 验证build/preprocess/neutralize/ic接口."""
import numpy as np
import pandas as pd
import pytest

from src.factors import pricevol, preprocess, neutralize, ic as icmod

rng = np.random.default_rng(7)
CODES = [f"sz.0000{i:02d}" for i in range(40)]
DATES = pd.bdate_range("2024-01-01", periods=80)


def make_panel() -> pd.DataFrame:
    rows = []
    for ci, c in enumerate(CODES):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DATES))))
        open_ = close * (1 + rng.normal(0, 0.005, len(DATES)))
        pre = np.r_[close[0], close[:-1]]
        for di, d in enumerate(DATES):
            suspended = (ci == 0) and (30 <= di < 50)  # 注入20日停牌
            rows.append({
                "code": c, "date": d, "open": open_[di],
                "high": max(open_[di], close[di]) * 1.01,
                "low": min(open_[di], close[di]) * 0.99,
                "close": close[di], "preclose": pre[di],
                "volume": 0.0 if suspended else rng.integers(1e5, 1e7),
                "amount": 1e6, "turn": rng.uniform(0.5, 5),
                "tradestatus": 0 if suspended else 1,
                "pctChg": (close[di] / pre[di] - 1) * 100,
                "close_badj": close[di], "tradable": not suspended,
                "mktcap": close[di] * rng.uniform(5e8, 5e9),
                "industry": "IND_A" if ci % 2 == 0 else "IND_B",
            })
    df = pd.DataFrame(rows)
    df["ln_mktcap"] = np.log(df["mktcap"])
    return df


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def test_pricevol_build(panel):
    fac = pricevol.build(panel)
    assert {"code", "date", "rev_5", "vol_20", "turn_mean"} <= set(fac.columns)
    # rev_5 第6日起应有值
    s = fac[fac["code"] == "sz.000001"]["rev_5"]
    assert s.iloc[6:].notna().mean() > 0.8
    # 停牌股(code0)在停牌段后短窗内应被护栏置NaN
    s0 = fac[(fac["code"] == "sz.000000") & (fac["date"] >= DATES[50])]
    assert s0["vol_20"].iloc[:15].isna().mean() > 0.5


def test_preprocess(panel):
    d = panel.assign(f=rng.normal(size=len(panel)))
    z = preprocess.xsec_zscore(preprocess.xsec_clip(d, ["f"]), ["f"])
    g = z.groupby("date")["f"].agg(["mean", "std"])
    assert np.allclose(g["mean"], 0, atol=1e-9)
    assert np.allclose(g["std"], 1, atol=0.05)


def test_neutralize_kills_size_beta(panel):
    # 构造与ln_mktcap完全线性相关的因子 -> size中性化残差应≈0
    d = panel.assign(f=panel["ln_mktcap"] * 2.0 + 1.0)
    out = neutralize.neutralize(d, ["f"], industry=False,
                                ind_categories=["IND_A", "IND_B"])
    res = out.loc[out["tradable"], "f"]
    assert res.abs().max() < 1e-8


def test_ic_direction(panel):
    d = icmod.add_fwd_ret(panel, [5])
    # fwd_ret_5(t) = close(t+5)/close(t)-1
    one = d[(d["code"] == "sz.000001") & (d["date"] == DATES[0])].iloc[0]
    exp = panel[(panel["code"] == "sz.000001")
                & (panel["date"] == DATES[5])]["close_badj"].iloc[0]
    assert one["fwd_ret_5"] == pytest.approx(
        exp / one["close_badj"] - 1, rel=1e-9)
    # 人为完美因子: f = fwd_ret 本身 -> IC≈1
    d["f_perfect"] = d["fwd_ret_5"]
    icv = icmod.daily_ic(d[d["tradable"]], "f_perfect", "fwd_ret_5",
                         mask_col="tradable")
    assert icv.mean() > 0.99
