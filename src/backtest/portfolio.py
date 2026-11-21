# -*- coding: utf-8 -*-
"""组合构建与持仓腿(legs)生成.

- 信号日t收盘pred五分位; t+1 open_badj买; 持有h日于open_badj卖
- 一字涨停放弃买入; 一字跌停顺延(<=10日, 仍限则末日强平并披露)
- 同组内续留个股不计双边成本(净额换手)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_groups(day: pd.DataFrame, pred_col: str = "pred",
                  n_groups: int = 5) -> pd.Series:
    """当日截面pred五分位 -> 组号1(最低)..5(最高/多头)."""
    s = day[pred_col]
    try:
        q = pd.qcut(s, n_groups, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=day.index)
    return q + 1


def build_legs(sig: pd.DataFrame, px: pd.DataFrame, calendar: np.ndarray,
               h: int, n_groups: int = 5,
               max_defer: int = 10) -> pd.DataFrame:
    """生成持仓腿长表.

    sig: [code,date(信号日),pred] 已按日五分位前的tradable子集
    px:  panel列 [code,date,open,high,low,pctChg,open_badj,close_badj,
                  preclose_badj,tradestatus]
    calendar: 全交易日历(np datetime64有序)
    返回列: [reb(信号日),group,code,entry,exit,buy_px,sell_px,ret,
             deferred,skip]
    """
    cal = np.asarray(calendar)
    pos = {d: i for i, d in enumerate(cal)}
    pxi = px.set_index(["code", "date"])
    legs = []
    for reb, day in sig.groupby("date"):
        ti = pos.get(reb)
        if ti is None or ti + 1 >= len(cal):
            continue
        e_date, x_sched = cal[ti + 1], cal[min(ti + 1 + h, len(cal) - 1)]
        day = day[day["pred"].notna()]
        day = day.assign(group=assign_groups(day))
        day = day[day["group"].notna()]
        for r in day.itertuples():
            e = pxi.loc[(r.code, e_date)] if (r.code, e_date) in pxi.index \
                else None
            if e is None or e["tradestatus"] != 1 \
                    or not np.isfinite(e["open_badj"]):
                legs.append({"reb": reb, "group": r.group, "code": r.code,
                             "entry": e_date, "exit": x_sched,
                             "buy_px": np.nan, "sell_px": np.nan,
                             "ret": np.nan, "deferred": 0, "skip": "suspend"})
                continue
            # 一字涨停: 开盘即封死挂不进 -> 放弃该笔(该份额留现金)
            if e["open"] == e["high"] == e["low"] and e["pctChg"] >= 9.9:
                legs.append({"reb": reb, "group": r.group, "code": r.code,
                             "entry": e_date, "exit": x_sched,
                             "buy_px": np.nan, "sell_px": np.nan,
                             "ret": np.nan, "deferred": 0, "skip": "limitup"})
                continue
            # 卖出日: 一字跌停卖不出 -> 逐日顺延直到非一字板(上限10日).
            # 顺延腿的exit越过下期entry时, 引擎按1/N权重与新仓同日记账
            xi, deferred = min(ti + 1 + h, len(cal) - 1), 0
            while deferred < max_defer:
                x = pxi.loc[(r.code, cal[xi])] \
                    if (r.code, cal[xi]) in pxi.index else None
                if x is not None and x["tradestatus"] == 1 \
                        and np.isfinite(x["open_badj"]):
                    if not (x["open"] == x["high"] == x["low"]
                            and x["pctChg"] <= -9.9):
                        break
                xi += 1
                deferred += 1
            x_date = cal[min(xi, len(cal) - 1)]
            x = pxi.loc[(r.code, x_date)] \
                if (r.code, x_date) in pxi.index else None
            if x is None or not np.isfinite(x["open_badj"]):
                sell_px = e["close_badj"]  # 极端兜底: 以末日收盘近似
            else:
                sell_px = x["open_badj"]
            legs.append({"reb": reb, "group": r.group, "code": r.code,
                         "entry": e_date, "exit": x_date,
                         "buy_px": e["open_badj"], "sell_px": sell_px,
                         "ret": sell_px / e["open_badj"] - 1,
                         "deferred": deferred, "skip": ""})
    return pd.DataFrame(legs)


def turnover_frac(cur_codes: set, prev_codes: set) -> tuple[float, float]:
    """(卖出比例, 买入比例): 等权近似 = 变更名单数/组内数."""
    if not prev_codes:
        return 0.0, 1.0
    keep = cur_codes & prev_codes
    n = max(len(prev_codes), 1)
    sell_f = 1 - len(keep) / n
    buy_f = 1 - len(keep) / max(len(cur_codes), 1)
    return sell_f, buy_f
