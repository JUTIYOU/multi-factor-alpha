# -*- coding: utf-8 -*-
"""PIT(时点可见)机制唯一入口.

一切"某字段在t日是否可见"的逻辑只能在这里实现:
- 财务/股本数据可用日 = pubDate; 缺失兜底 statDate+90d 并打 imputed 标记
- asof_join 对所有业务代码提供统一的 (code, date) -> 最近已披露记录 映射

业务代码(factors/backtest)禁止自己处理公告日或向前填充财务数据.
"""
from __future__ import annotations

import pandas as pd

IMPUTE_DAYS = 90  # pubDate缺失时的保守披露滞后(介于季报法定上限与实用之间)


def attach_available_date(fin: pd.DataFrame) -> pd.DataFrame:
    """财务原始表(含code,pubDate,statDate) -> 加 available_date / imputed.

    available_date = pubDate; pubDate缺失 -> statDate + IMPUTE_DAYS, imputed=True.
    """
    out = fin.copy()
    out["pubDate"] = pd.to_datetime(out["pubDate"], errors="coerce")
    out["statDate"] = pd.to_datetime(out["statDate"], errors="coerce")
    miss = out["pubDate"].isna()
    out["available_date"] = out["pubDate"].fillna(
        out["statDate"] + pd.Timedelta(days=IMPUTE_DAYS))
    out["imputed"] = miss
    return out


def asof_join(panel: pd.DataFrame, fin: pd.DataFrame,
              fields: list[str], suffix: str = "fin_") -> pd.DataFrame:
    """对panel每行(code,date)取 available_date<=date 的最新fin记录.

    返回panel + {suffix}{field} + _fin_available_date/_fin_statDate/_fin_imputed.
    merge_asof 保证只向后匹配; 再做显式断言双保险.
    """
    p = panel.sort_values("date", kind="mergesort")
    f = (fin.sort_values("available_date", kind="mergesort")
            [["code", "statDate", "available_date", "imputed", *fields]])
    m = pd.merge_asof(p, f, left_on="date", right_on="available_date",
                      by="code", direction="backward")
    ok = m["available_date"].notna()
    assert (m.loc[ok, "available_date"] <= m.loc[ok, "date"]).all(), \
        "PIT violation: available_date > date"
    m = m.rename(columns={
        "available_date": "_fin_available_date",
        "statDate": "_fin_statDate",
        "imputed": "_fin_imputed",
        **{c: f"{suffix}{c}" for c in fields},
    })
    return m


def pit_check(joined: pd.DataFrame, date_col: str = "date") -> bool:
    """对已asof_join过的表断言: 非空的_fin_available_date <= date."""
    ok = joined["_fin_available_date"].notna()
    return bool((joined.loc[ok, "_fin_available_date"]
                 <= joined.loc[ok, date_col]).all())
