# -*- coding: utf-8 -*-
"""T0.2 BaoStock冒烟测试: 验证关键接口能力与字段名.
运行: .venv/Scripts/python scripts/smoke_baostock.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baostock as bs
from src.data.baostock_client import Session, rs_to_df

FIELDS_DAILY = ("date,code,open,high,low,close,preclose,volume,amount,"
                "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST")


def main():
    report = []

    with Session():
        # 1. 交易日历
        df = rs_to_df(bs.query_trade_dates(start_date="2018-01-01", end_date="2018-01-31"))
        report.append(f"[日历] 2018-01 记录数={len(df)}, 列={list(df.columns)}")
        assert "is_trading_day" in df.columns

        # 2. 中证500历史成分(关键验证: date参数是否真实有效)
        df_now = rs_to_df(bs.query_zz500_stocks())
        df_old = rs_to_df(bs.query_zz500_stocks(date="2018-01-31"))
        same = set(df_now["code"]) == set(df_old["code"])
        report.append(f"[成分] 当前={len(df_now)}只, 2018-01-31={len(df_old)}只, "
                      f"两期完全相同={same}, 列={list(df_old.columns)}")
        report.append(f"       df_old updateDate取值={sorted(df_old['updateDate'].unique())[:3]}...")

        # 3. 日线全字段(不复权)
        df = rs_to_df(bs.query_history_k_data_plus(
            "sh.600000", FIELDS_DAILY,
            start_date="2018-01-02", end_date="2018-01-10",
            frequency="d", adjustflag="3"))
        report.append(f"[日线] sh.600000 行数={len(df)}, 列={list(df.columns)}")

        # 4. 复权因子表
        df = rs_to_df(bs.query_adjust_factor(
            code="sh.600000", start_date="2017-01-01", end_date="2020-01-01"))
        report.append(f"[复权因子] 行数={len(df)}, 列={list(df.columns)}")

        # 5. 指数日线
        df = rs_to_df(bs.query_history_k_data_plus(
            "sh.000905", "date,code,close,pctChg",
            start_date="2018-01-02", end_date="2018-01-10",
            frequency="d", adjustflag="3"))
        report.append(f"[指数] sh.000905 行数={len(df)}")

        # 6. 财务单点(验证pubDate/statDate字段名)
        df = rs_to_df(bs.query_profit_data(code="sh.600000", year=2018, quarter=1))
        report.append(f"[财务profit] 行数={len(df)}, 列={list(df.columns)}")
        assert "pubDate" in df.columns and "statDate" in df.columns

        df = rs_to_df(bs.query_growth_data(code="sh.600000", year=2018, quarter=1))
        report.append(f"[财务growth] 行数={len(df)}, 列={list(df.columns)}")
        df = rs_to_df(bs.query_balance_data(code="sh.600000", year=2018, quarter=1))
        report.append(f"[财务balance] 行数={len(df)}, 列={list(df.columns)}")
        df = rs_to_df(bs.query_cash_flow_data(code="sh.600000", year=2018, quarter=1))
        report.append(f"[财务cash_flow] 行数={len(df)}, 列={list(df.columns)}")

        # 7. stock_basic / industry
        df = rs_to_df(bs.query_stock_basic(code="sh.600000"))
        report.append(f"[basic] 列={list(df.columns)}")
        df = rs_to_df(bs.query_stock_industry(code="sh.600000"))
        report.append(f"[行业] 列={list(df.columns)}")

    out = "\n".join(report)
    print(out)
    Path("reports/smoke_baostock.log").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
