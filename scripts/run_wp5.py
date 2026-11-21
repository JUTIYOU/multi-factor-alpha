# -*- coding: utf-8 -*-
"""WP5 编排: 含成本五分位分层回测 + 四段风险归因.

口径: t收盘信号 -> t+1 open_badj买 -> t+1+h open_badj卖; 买5bp/卖15bp;
      一字涨停弃买,一字跌停顺延; 续留免费. 只报告含成本净值(门禁).
矩阵: {P3,P4} x h{5,10}. 基准: sh.000905.
产出: reports/backtest_summary.csv / backtest_nav.csv /
      backtest_layers.csv / backtest_attribution.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import costs, portfolio as pf, engine, attribution
from src.data.baostock_client import load_yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"


def main() -> None:
    if not (PROC / "industry_map.parquet").exists():
        sys.exit("先跑 scripts/build_industry_map.py")
    cfg = load_yaml(ROOT / "configs" / "cost.yaml")
    c = costs.load_costs(cfg)
    print(f"[wp5] costs buy={c['buy_bp']}bp sell={c['sell_bp']}bp",
          flush=True)
    pr = pd.read_parquet(PROC / "wp3_preds.parquet")
    px = pd.read_parquet(
        PROC / "panel.parquet",
        columns=["code", "date", "open", "high", "low", "pctChg",
                 "tradestatus", "open_badj", "close_badj",
                 "preclose_badj", "ln_mktcap"])
    ind = pd.read_parquet(PROC / "industry_map.parquet")
    cal = np.sort(px["date"].unique())
    # 信号日 -> code级 行业/市值映射(PIT)
    px_ci = px.set_index(["date", "code"])
    ind_ci = ind.set_index(["date", "code"])["industry"]

    bench = pd.read_parquet(RAW / "index_quotes.parquet")
    bench = bench[bench.code == "sh.000905"].copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench["close"] = bench["close"].astype(float)
    bench = bench.set_index("date")["close"]

    summary, navs, layers, attrs = [], [], [], []
    for (h, proto), g in pr.groupby(["horizon", "protocol"]):
        t0 = time.time()
        sig = g[g["tradable"].fillna(False)][["code", "date", "pred"]]
        test_dates = np.sort(sig["date"].unique())
        # 每h个交易日调仓
        pos = {d: i for i, d in enumerate(cal)}
        idxs = sorted(pos[d] for d in test_dates if d in pos)
        reb_dates = [cal[i] for i in idxs[::h]]
        sig = sig[sig["date"].isin(reb_dates)]
        legs = pf.build_legs(sig, px, cal, h, n_groups=5)
        skip_rate = (legs["skip"] != "").mean()
        daily = engine.daily_group_returns(legs, px, cal)
        nav = engine.nav_with_costs(daily, legs, c["buy_bp"],
                                    c["sell_bp"], group=5)
        m = engine.metrics(nav)
        m.update(protocol=proto, horizon=h,
                 skip_rate=round(skip_rate, 4),
                 n_reb=len(reb_dates), secs=round(time.time() - t0, 1))
        # 基准对齐
        b = bench.reindex(nav["date"]).ffill()
        m["bench_total"] = float(b.iloc[-1] / b.iloc[0] - 1)
        summary.append(m)
        navs.append(nav.assign(protocol=proto, horizon=h))
        # 分层: 各组期收益累计
        per = (legs[legs["ret"].notna()]
               .groupby(["reb", "group"])["ret"].mean().unstack())
        cum = (1 + per).cumprod() - 1
        cum.index.name = "reb"
        layers.append(cum.reset_index().assign(
            protocol=proto, horizon=h))
        # 归因: reb -> code级Series
        ind_by_reb, size_by_reb = {}, {}
        for r in reb_dates:
            try:
                ind_by_reb[r] = ind_ci.loc[r]
                size_by_reb[r] = px_ci.loc[r]["ln_mktcap"]
            except KeyError:
                continue
        a = attribution.attribute_all(legs, ind_by_reb, size_by_reb,
                                      group=5)
        a = a.assign(protocol=proto, horizon=h)
        attrs.append(a)
        print(f"[wp5] {proto} h={h}: net={m['total_ret']:.3f} "
              f"ann={m['ann_ret']:.3f} sharpe={m['sharpe']:.2f} "
              f"maxdd={m['maxdd']:.3f} skip={skip_rate:.3f} "
              f"({m['secs']}s)", flush=True)

    pd.DataFrame(summary).to_csv(REPORTS / "backtest_summary.csv",
                                 index=False)
    pd.concat(navs).to_csv(REPORTS / "backtest_nav.csv", index=False)
    pd.concat(layers).to_csv(REPORTS / "backtest_layers.csv",
                             index=False)
    pd.concat(attrs).to_csv(REPORTS / "backtest_attribution.csv",
                            index=False)
    print("[wp5] ALL DONE -> reports/backtest_*.csv", flush=True)


if __name__ == "__main__":
    main()
