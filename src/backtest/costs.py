# -*- coding: utf-8 -*-
"""成本与成交规则(configs/cost.yaml口径).

买5bp(佣3+滑2), 卖15bp(佣3+印10+滑2). 一字板判定: open==high==low且|pctChg|>=9.9%.
"""
from __future__ import annotations

import pandas as pd


def load_costs(cfg: dict) -> dict:
    c = cfg["cost"]
    return {
        "buy_bp": c["commission_bp"] + c["slippage_bp"],
        "sell_bp": c["commission_bp"] + c["stamp_bp"] + c["slippage_bp"],
        "limit_rule": cfg["execution"].get("limit_rule", "skip_if_limit"),
    }


def one_word_limit(row: pd.DataFrame, up: bool) -> pd.Series:
    """一字涨跌停: 当日只有一个成交价且触板."""
    flat = (row["open"] == row["high"]) & (row["high"] == row["low"])
    thr = 9.9
    return flat & (row["pctChg"] >= thr) if up else flat & (
        row["pctChg"] <= -thr)
