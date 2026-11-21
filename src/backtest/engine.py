# -*- coding: utf-8 -*-
"""日频净值引擎: 持仓腿 -> 逐日组收益 -> 含成本净值.

腿内逐日收益: 入场日 close/open-1; 出场日 open/preclose-1; 中间日 close/preclose-1.
调仓边界日 = 旧仓exit日 = 新仓entry日(同一开盘): 日收益 = 旧腿ret + 新腿ret
(顺序复利近似); 双边成本(卖旧+买新)同日记扣. 续留个股不产生该日腿切换,免费.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_group_returns(legs: pd.DataFrame, px: pd.DataFrame,
                        calendar: np.ndarray) -> pd.DataFrame:
    """持仓腿展开为逐日行 [reb, group, date, ret]."""
    cal = np.asarray(calendar)
    pos = {d: i for i, d in enumerate(cal)}
    ok = legs[legs["ret"].notna()].copy()
    ok["ei"] = ok["entry"].map(pos).astype(int)
    ok["xi"] = ok["exit"].map(pos).astype(int)
    exp = []
    for leg in ok.itertuples():
        for i in range(leg.ei, leg.xi + 1):
            exp.append((leg.reb, leg.group, leg.code, cal[i],
                        i == leg.ei, i == leg.xi))
    e = pd.DataFrame(exp, columns=["reb", "group", "code", "date",
                                   "is_entry", "is_exit"])
    e = e.merge(px[["code", "date", "open_badj", "close_badj",
                    "preclose_badj"]], on=["code", "date"], how="left")
    e["ret"] = np.select(
        [e["is_entry"], e["is_exit"]],
        [e["close_badj"] / e["open_badj"] - 1,
         e["open_badj"] / e["preclose_badj"] - 1],
        default=e["close_badj"] / e["preclose_badj"] - 1)
    # 组合日收益 = 各腿收益按固定权重1/N求和(不是在场腿均值:
    # 顺延腿延长段只剩零星股票,均值会放大单日为全组合收益)
    n_per = (e.groupby(["reb", "group"])["code"].nunique()
             .rename("n").reset_index())
    e = e.merge(n_per, on=["reb", "group"])
    e["wret"] = e["ret"] / e["n"]
    return (e.groupby(["reb", "group", "date"])["wret"].sum()
            .rename("ret").reset_index())


def nav_with_costs(daily: pd.DataFrame, legs: pd.DataFrame,
                   buy_bp: float, sell_bp: float,
                   group: int = 5) -> pd.DataFrame:
    """组日收益 -> 净值(边界日记双边成本, 续留免费)."""
    dg = daily[daily["group"] == group]
    # 同日可能有两个reb的行(调仓边界日): 旧仓open/preclose + 新仓close/open
    # 顺序发生 -> 相加(≈顺序复利); 非边界日只有一行,sum=自身
    day_ret = dg.groupby("date")["ret"].sum()
    gross = (1 + day_ret.fillna(0)).cumprod()

    legs_g = legs[(legs["group"] == group) & legs["ret"].notna()]
    rebs = sorted(legs_g["reb"].unique())
    entry_by_reb = legs_g.groupby("reb")["entry"].first()
    cost_by_date: dict = {}
    prev: set = set()
    for reb in rebs:
        cur = set(legs_g[legs_g["reb"] == reb]["code"])
        keep = cur & prev
        sell_f = 1 - len(keep) / max(len(prev), 1) if prev else 0.0
        buy_f = 1 - len(keep) / max(len(cur), 1)
        b = entry_by_reb[reb]
        cost_by_date[b] = cost_by_date.get(b, 0.0) + (
            buy_f * buy_bp + sell_f * sell_bp) / 1e4
        prev = cur
    net = day_ret.copy()
    for dt, c in cost_by_date.items():
        if dt in net.index:
            net.loc[dt] -= c
    nav = (1 + net.fillna(0)).cumprod()
    return pd.DataFrame({"date": nav.index,
                         "nav_net": nav.values,
                         "nav_gross": gross.values,
                         "ret_net": net.fillna(0).values,
                         "ret_gross": day_ret.fillna(0).values,
                         "turnover_ann": _annual_turnover(legs_g)})


def _annual_turnover(legs_g: pd.DataFrame) -> float:
    """年均单边换手 ≈ 每期变更率 × 期数/年."""
    rebs = sorted(legs_g["reb"].unique())
    if len(rebs) < 2:
        return np.nan
    prev = None
    chg = []
    for r in rebs:
        cur = set(legs_g[legs_g["reb"] == r]["code"])
        if prev is not None:
            chg.append(1 - len(cur & prev) / max(len(prev), 1))
        prev = cur
    yrs = (rebs[-1] - rebs[0]) / np.timedelta64(365, "D")
    return float(np.mean(chg) * len(chg) / max(yrs, 1e-9))


def metrics(nav: pd.DataFrame, rf: float = 0.0) -> dict:
    r = nav["ret_net"]
    n = max(len(r), 1)
    return {
        "ann_ret": nav["nav_net"].iloc[-1] ** (252 / n) - 1,
        "sharpe": (r.mean() - rf / 252) / r.std() * np.sqrt(252)
        if r.std() > 0 else np.nan,
        "maxdd": float((nav["nav_net"]
                        / nav["nav_net"].cummax() - 1).min()),
        "total_ret": nav["nav_net"].iloc[-1] - 1,
        "turnover_ann": nav["turnover_ann"].iloc[0],
        "gross_ann": nav["nav_gross"].iloc[-1] ** (252 / n) - 1,
        "cost_drag_ann": (nav["nav_gross"].iloc[-1]
                          / nav["nav_net"].iloc[-1]) ** (252 / n) - 1,
    }
