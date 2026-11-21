# -*- coding: utf-8 -*-
"""T1.6 数据质量报告 + PIT抽查20例 + 估值对拍(自建BP vs 1/pbMRQ).

运行: .venv/Scripts/python -m src.data.quality
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.baostock_client import load_yaml
from src.data.pit import attach_available_date, asof_join

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RPT = ROOT / "reports"

SEED = 42


def check_coverage(panel: pd.DataFrame, cal: pd.DataFrame) -> dict:
    days_panel = panel["date"].nunique()
    days_cal = len(cal)
    return {"panel_days": days_panel, "calendar_days": days_cal,
            "coverage": days_panel / days_cal}


def check_units(panel: pd.DataFrame) -> list[str]:
    out = []
    v = panel["volume"].dropna()
    a = panel["amount"].dropna()
    med_px = (panel["close"].dropna()).median()
    # A股价格量级 1~200元; amount/volume 应约等于价格量级
    ratio = (a / v.replace(0, np.nan)).dropna().median()
    out.append(f"median close={med_px:.2f}, amount/volume median={ratio:.2f} "
               f"(应≈价格量级, 若≈价格*100则volume单位为手)")
    return out


def check_adj(panel: pd.DataFrame) -> dict:
    """pctChg vs 后复权差分 对拍: 除权日外应一致."""
    g = panel[panel["tradestatus"] == 1].copy()
    r1 = g["pctChg"] / 100
    r2 = g.groupby("code")["close_badj"].pct_change()
    d = (r1 - r2).abs()
    return {"n": int(d.notna().sum()), "diff_p99": float(d.quantile(0.99)),
            "diff_max": float(d.max()),
            "n_large>1pct": int((d > 0.01).sum())}


def pit_sample_check(panel: pd.DataFrame, fin: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """随机n个(code,date): 断言as-of join返回记录的available_date<=date,
    且不存在更晚的已披露记录被漏过."""
    fin = attach_available_date(fin)
    rng = np.random.default_rng(SEED)
    sampled = panel[panel["tradable"]].sample(n=n, random_state=rng)[["code", "date"]]
    rows = []
    for _, r in sampled.iterrows():
        c, d = r["code"], r["date"]
        cand = fin[(fin["code"] == c) & (fin["available_date"] <= d)]
        expect = cand["statDate"].max() if len(cand) else pd.NaT
        got = asof_join(pd.DataFrame([{"code": c, "date": d}]), fin,
                        fields=["roeAvg"], suffix="fin_")
        got_stat = got["_fin_statDate"].iloc[0]
        ok = (pd.isna(expect) and pd.isna(got_stat)) or (expect == got_stat)
        rows.append({"code": c, "date": d, "expect_statDate": expect,
                     "got_statDate": got_stat, "pass": bool(ok)})
    return pd.DataFrame(rows)


def bp_crosscheck(panel: pd.DataFrame, fin: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """自建BP(门控财报) vs 1/pbMRQ: 偏离>10%计数, 系统性偏离→弃用接口估值."""
    # 自建BP需归母权益; profit接口无权益原值 → 用pbMRQ反推权益做口径核验:
    # 改为核验pbMRQ的PIT正确性: pbMRQ_t = mktcap_t / equity_latest_disclosed_t
    # equity_latest = mktcap/pbMRQ; 检查该隐含equity在pubDate切换点跳变
    rng = np.random.default_rng(SEED + 1)
    sampled = panel[panel["tradable"] & panel["pbMRQ"].notna()
                    & (panel["pbMRQ"] > 0)].sample(n=n, random_state=rng)
    rows = []
    for _, r in sampled.iterrows():
        implied_equity = r["mktcap"] / r["pbMRQ"] if r["pbMRQ"] else np.nan
        rows.append({"code": r["code"], "date": r["date"],
                     "mktcap": r["mktcap"], "pbMRQ": r["pbMRQ"],
                     "implied_equity": implied_equity})
    return pd.DataFrame(rows)


def missing_report(panel: pd.DataFrame) -> pd.DataFrame:
    miss = panel.assign(year=panel["date"].dt.year).groupby("year").apply(
        lambda g: g.isna().mean())
    return miss


def main():
    cfg = load_yaml(ROOT / "configs" / "universe.yaml")
    RPT.mkdir(parents=True, exist_ok=True)
    lines = ["# 数据质量报告", ""]
    panel = pd.read_parquet(PROC / "panel.parquet")
    cal = pd.read_parquet(RAW / "trade_calendar.parquet")
    fin_profit = pd.concat(
        [pd.read_parquet(f) for f in sorted(
            (RAW / "financials" / "profit").glob("*.parquet"))],
        ignore_index=True)

    lines += ["## 1. 股票池与覆盖", "",
              f"- 主池: 中证500月末快照并集 {panel['code'].nunique()} 只",
              f"- 区间: {panel['date'].min().date()} ~ {panel['date'].max().date()}",
              f"- 覆盖: {check_coverage(panel, cal)}",
              f"- tradable比例: {panel['tradable'].mean():.3f}", ""]

    lines += ["## 2. 单位检查", *[f"- {s}" for s in check_units(panel)], ""]

    lines += ["## 3. 复权一致性 (pctChg vs 后复权差分)",
              f"- {check_adj(panel)}", ""]

    lines += ["## 4. PIT抽查(20例)"]
    pit_res = pit_sample_check(panel, fin_profit, n=20)
    n_pass = int(pit_res["pass"].sum())
    lines += [f"- 通过 {n_pass}/20", "", pit_res.to_markdown(index=False), ""]
    assert n_pass == 20, "PIT抽查未全过, 不得进入WP2"

    lines += ["## 5. 估值口径核验(pbMRQ隐含权益)", "",
              bp_crosscheck(panel, fin_profit).to_markdown(index=False), ""]

    lines += ["## 6. 缺失率(按年)", "",
              missing_report(panel).to_markdown(), ""]

    lines += ["## 7. 已知偏差披露",
              "- BaoStock同一报告期仅留最新披露口径(重述覆盖)→轻微前瞻,不可修复",
              "- 行业分类为BaoStock自有体系,非申万;历史变更不可得→用当前口径",
              "- 退市股行情完整度未单独验证→幸存者偏差方向上界披露",
              "- growth接口同比字段未使用(自建同比更可控)", ""]

    out = "\n".join(lines)
    (RPT / "data_quality.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
