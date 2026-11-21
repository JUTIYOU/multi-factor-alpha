# -*- coding: utf-8 -*-
"""T1.4/T1.5 面板构建: 合并行情+复权价+市值(财报门控股本)+tradable掩膜.

运行: .venv/Scripts/python -m src.data.panel
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

NUM_COLS = ["open", "high", "low", "close", "preclose", "volume", "amount",
            "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]


def load_quotes() -> pd.DataFrame:
    files = sorted((RAW / "quotes").glob("*.parquet"))
    frames, corrupt = [], []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            corrupt.append(f.stem)  # 记录待重拉,不阻塞面板构建
    if corrupt:
        pd.Series(corrupt, name="code").to_csv(
            RAW / "quotes" / "_corrupt.csv", index=False)
        print(f"  WARN corrupt parquet skipped: {len(corrupt)} "
              f"-> data/raw/quotes/_corrupt.csv", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["tradestatus", "isST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def load_adj() -> pd.DataFrame:
    files = sorted((RAW / "adj_factor").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["code", "dividOperateDate",
                                     "backAdjustFactor", "foreAdjustFactor"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["dividOperateDate"] = pd.to_datetime(df["dividOperateDate"])
    for c in ["backAdjustFactor", "foreAdjustFactor", "adjustFactor"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_badj(quotes: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """后复权价 = raw_close * BF_asof(t).

    BF(backAdjustFactor)是BaoStock的**累计**因子(已验证:相邻事件行比值=
    close_{e-1}/preclose_e), 故直接as-of取最近一次事件的BF, 不可再cumprod.
    首个事件前的日期: 基线 = BF(首事件)/真实单事件因子(首事件).
    """
    df = quotes.sort_values("date", kind="mergesort").copy()
    # 每事件真实单期因子 f_e = 前收close/除权日preclose
    prev_close = (quotes.sort_values(["code", "date"])
                  .groupby("code")["close"].shift(1))
    qf = quotes.assign(prev_close=prev_close)[
        ["code", "date", "preclose", "prev_close"]]
    ev = (adj[["code", "dividOperateDate", "backAdjustFactor"]]
          .dropna()
          .rename(columns={"dividOperateDate": "date"})
          .merge(qf, on=["code", "date"], how="left"))
    ev["f_true"] = ev["prev_close"] / ev["preclose"]
    ev["bf_use"] = ev["backAdjustFactor"]
    # 基线行: 放在首事件远早日期, 覆盖首事件之前的所有t
    first = (ev.sort_values(["code", "date"]).groupby("code").head(1)
             .copy())
    first["bf_use"] = first["backAdjustFactor"] / first["f_true"]
    first["date"] = first["date"] - pd.Timedelta(days=3650)
    allf = (pd.concat([ev[["code", "date", "bf_use"]],
                       first[["code", "date", "bf_use"]]])
            .sort_values("date", kind="mergesort"))
    if allf.empty:
        df["badj_cum"] = 1.0  # 无除权事件: 后复权=原价
    else:
        df = pd.merge_asof(df, allf, on="date", by="code",
                           direction="backward")
        df["badj_cum"] = df["bf_use"].fillna(1.0)
    for col in ["open", "high", "low", "close"]:
        df[f"{col}_badj"] = df[col] * df["badj_cum"]
    df = df.sort_values(["code", "date"], kind="mergesort")
    df["preclose_badj"] = df.groupby("code")["close_badj"].shift(1)
    return df.drop(columns=["bf_use"], errors="ignore")


def build_membership(snaps: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """月末成分快照 -> 日度成员表(code,date).

    口径: 快照日s的成分自s(含)生效, 到下一个全局快照日s'(不含)失效.
    缺席中间快照的股票不会错误延续成员身份.
    """
    s = snaps.copy()
    s["snapshot_date"] = pd.to_datetime(s["snapshot_date"])
    snap_dates = np.array(sorted(s["snapshot_date"].unique()))
    dates = np.sort(calendar["date"].values)
    far = np.datetime64("2100-01-01")

    segs = []
    for i, sd in enumerate(snap_dates):
        sd_next = snap_dates[i + 1] if i + 1 < len(snap_dates) else far
        codes = s.loc[s["snapshot_date"] == sd, "code"].unique()
        segs.append(pd.DataFrame({"code": codes, "eff_start": sd,
                                  "eff_end": sd_next}))
    seg = pd.concat(segs, ignore_index=True)

    rows = []
    for code, st, en in seg.itertuples(index=False):
        in_range = dates[(dates >= np.datetime64(st)) & (dates < np.datetime64(en))]
        rows.append(pd.DataFrame({"code": code, "date": in_range}))
    member = pd.concat(rows, ignore_index=True).drop_duplicates()
    member["date"] = pd.to_datetime(member["date"])
    return member


def build_mask(panel: pd.DataFrame, member: pd.DataFrame,
               basic: pd.DataFrame, min_list_days: int) -> pd.Series:
    ipo = basic.set_index("code")["ipoDate"]
    ipo = pd.to_datetime(ipo, errors="coerce")
    list_days = (panel["date"] - panel["code"].map(ipo)).dt.days
    in_uni = panel.set_index(["code", "date"]).index.isin(
        member.set_index(["code", "date"]).index)
    mask = (pd.Series(in_uni, index=panel.index)
            & (panel["tradestatus"] == 1)
            & (panel["isST"].fillna(0) == 0)
            & (list_days >= min_list_days))
    return mask.fillna(False)


def attach_mktcap(panel: pd.DataFrame, fin_profit: pd.DataFrame) -> pd.DataFrame:
    """日市值 = close * 截至t日已披露totalShare (pubDate门控)."""
    fin = attach_available_date(fin_profit.dropna(subset=["statDate"]))
    m = asof_join(panel, fin, fields=["totalShare", "liqaShare"], suffix="fin_")
    m["mktcap"] = m["close"] * m["fin_totalShare"]
    m["float_mktcap"] = m["close"] * m["fin_liqaShare"]
    m["ln_mktcap"] = np.log(m["mktcap"].where(m["mktcap"] > 0))
    return m


def load_fin_profit() -> pd.DataFrame:
    files = sorted((RAW / "financials" / "profit").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # BaoStock返回全字符串, 非键字段统一转数值
    for c in df.columns:
        if c not in ("code", "pubDate", "statDate"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    cfg = load_yaml(ROOT / "configs" / "universe.yaml")
    PROC.mkdir(parents=True, exist_ok=True)

    print("[1/5] load quotes", flush=True)
    q = load_quotes()
    assert not q.duplicated(["code", "date"]).any(), "duplicate (code,date)"

    print("[2/5] badj prices", flush=True)
    q = add_badj(q, load_adj())

    print("[3/5] membership + mask", flush=True)
    cal = pd.read_parquet(RAW / "trade_calendar.parquet")
    snaps = pd.read_parquet(RAW / "zz500_snapshots.parquet")
    member = build_membership(snaps, cal)
    basic = pd.read_parquet(RAW / "stock_basic_universe.parquet")
    q["tradable"] = build_mask(q, member, basic, cfg["filters"]["min_list_days"])

    print("[4/5] mktcap (pubDate-gated shares)", flush=True)
    q = attach_mktcap(q, load_fin_profit())

    print("[5/5] save", flush=True)
    q = q.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    q.to_parquet(PROC / "panel.parquet", index=False)
    print(f"panel rows={len(q)}, codes={q['code'].nunique()}, "
          f"tradable_rate={q['tradable'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
