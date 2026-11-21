# -*- coding: utf-8 -*-
"""导出dashboard部署用小数据包 -> reports/demo/*.csv.

从 data/processed(不上传) 聚合出KB级CSV, dashboard只读这些:
- index_nav.csv     基准净值(与回测日期对齐)
- tradable_daily.csv 每日可交易股票数
- roll_ic.csv       合成信号逐日IC -> 63日滚动均值(4臂)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO, PROC = ROOT / "reports" / "demo", ROOT / "data" / "processed"
DEMO.mkdir(exist_ok=True)

# ---- 1) 基准净值 ----
nav = pd.read_csv(ROOT / "reports" / "backtest_nav.csv",
                  parse_dates=["date"])
idx = pd.read_parquet(ROOT / "data" / "raw" / "index_quotes.parquet")
idx["close"] = pd.to_numeric(idx["close"])
idx["date"] = pd.to_datetime(idx["date"])
dates = sorted(nav.date.unique())
b = idx.set_index("date")["close"].reindex(dates).ffill()
pd.DataFrame({"date": dates, "bench_nav": (b / b.iloc[0]).values}
             ).to_csv(DEMO / "index_nav.csv", index=False)
print("index_nav:", len(dates))

# ---- 2) 每日可交易数 ----
p = pd.read_parquet(PROC / "panel.parquet",
                    columns=["code", "date", "tradable"])
tc = (p[p.tradable.fillna(False)].groupby("date").size()
      .rename("n_tradable").reset_index())
tc.to_csv(DEMO / "tradable_daily.csv", index=False)
print("tradable_daily:", len(tc), f"{tc.n_tradable.min()}-"
      f"{tc.n_tradable.max()}")

# ---- 3) 合成信号63日滚动IC ----
preds = pd.read_parquet(PROC / "wp3_preds.parquet")
out = []
for (proto, hz), g in preds.groupby(["protocol", "horizon"]):
    y = f"fwd_ret_{hz}"
    t = g[g.tradable.fillna(False) & g.pred.notna() & g[y].notna()]
    rk = t.groupby("date")[[ "pred", y]].rank(pct=True)
    rk["date"] = t.date.values
    ic = rk.groupby("date").apply(
        lambda x: x["pred"].corr(x[y]), include_groups=False)
    roll = ic.rolling(63, min_periods=30).mean().rename("roll_ic")
    tmp = roll.reset_index().rename(columns={"index": "date"})
    tmp["protocol"], tmp["horizon"] = proto, hz
    out.append(tmp[["date", "roll_ic", "protocol", "horizon"]])
    print(f"roll_ic {proto} h{hz}: {ic.notna().sum()} days")
pd.concat(out).to_csv(DEMO / "roll_ic.csv", index=False)
print("-> reports/demo/")
