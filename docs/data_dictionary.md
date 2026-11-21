# 数据字典 v1.0

> 口径断言：**任一因子在 t 日收盘时用到的最晚信息日期、来源接口、泄漏可能** 均可在此追溯。

## 1. 行情面板 `data/processed/panel.parquet`

主键 (code, date)。行=股票×交易日（含停牌日，停牌日 OHLC=前收、volume=0、tradestatus=0）。

| 字段 | 类型 | 来源接口 | 单位/口径 | t日最晚信息 |
|---|---|---|---|---|
| code | str | — | sh./sz.+6位 | — |
| date | datetime | — | 交易日 | — |
| open/high/low/close | float | query_history_k_data_plus(adj=3不复权) | 元 | t收盘 |
| preclose | float | 同上 | 元(除权前收) | t收盘 |
| volume | float | 同上 | 股 | t收盘 |
| amount | float | 同上 | 元 | t收盘 |
| turn | float | 同上 | %(停牌为NaN) | t收盘 |
| tradestatus | int | 同上 | 1正常/0停牌 | t收盘 |
| pctChg | float | 同上 | %(用÷100) | t收盘 |
| peTTM/pbMRQ/psTTM/pcfNcfTTM | float | 同上 | 接口口径(T1.6对拍核验) | t收盘* |
| isST | int | 同上 | 1/0 | t收盘 |
| open_badj…close_badj | float | 本地: raw×cumprod(backFactor≤t) | 后复权元 | t收盘 |
| preclose_badj | float | 本地: close_badj昨日值 | 后复权元 | t收盘 |
| mktcap | float | close×fin_totalShare(pubDate门控) | 元 | t收盘+最近pubDate |
| float_mktcap | float | close×fin_liqaShare | 元 | 同上 |
| ln_mktcap | float | ln(mktcap) | — | 同上 |
| tradable | bool | 本地合成 | in_universe&非停牌&非ST&上市≥60日 | t收盘 |

\* peTTM 等的 PIT 正确性依赖 BaoStock 内部口径，T1.6 用自建门控 BP 对拍 20 例核验，系统性偏离则弃用改自建。

## 2. 原始层 `data/raw/`

| 文件 | 来源 | 说明 |
|---|---|---|
| trade_calendar.parquet | query_trade_dates | 交易日历(is_trading_day=1) |
| zz500_snapshots.parquet | query_zz500_stocks(date=月末) | (updateDate,code,code_name,snapshot_date) |
| universe_codes.csv | 上者并集 | 1217只 |
| stock_basic*.parquet | query_stock_basic | ipoDate/outDate/type/status |
| industry.parquet | query_stock_industry | 当前口径行业 |
| industry_history.parquet | query_stock_industry(code,date=半年末) | PIT行业(分段有效) |
| index_quotes.parquet | 指数日线 sh.000905 | 基准 |
| quotes/{code}.parquet | 日线不复权 | per-code断点续传 |
| adj_factor/{code}.parquet | query_adjust_factor | dividOperateDate/fore/backAdjustFactor |
| financials/{profit,balance,cash_flow}/{code}.parquet | 季频财务 | pubDate/statDate+比率字段 |

## 3. 财务字段（季频，available_date=pubDate）

| 接口 | 关键字段 | 口径 | 最晚信息 |
|---|---|---|---|
| profit | roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare | netProfit/MBRevenue=元;股本=股 | pubDate |
| balance | liabilityToAsset,assetToEquity,currentRatio,quickRatio | 比率 | pubDate |
| cash_flow | CFOToNP,CFOToOR,CFOToGr,CAToAsset,NCAToAsset | 比率(累计口径) | pubDate |

**PIT规则**: `available_date=pubDate`；pubDate缺失→`statDate+90d`且`imputed=True`。
流量字段为报告期累计，单季/TTM须差分且各期均须满足pubDate门控。

## 4. 已知偏差（披露）

1. BaoStock同报告期仅留最新披露口径（重述覆盖）→固有轻微前瞻，不可修复。
2. 行业历史为半年度网格近似(6/12月末)，非逐日变更表。
3. 退市股行情完整度未经第三方对拍→幸存者偏差方向上界。
4. 无Tushare→估值对拍改为自建门控BP vs 接口pbMRQ互验。
5. growth接口未使用（自建同比，去年同期同样门控）。


## ��֪��������
- `index_quotes.parquet`: BaoStockԭʼ�ַ���δת��, ���Ѷ���`pd.to_datetime(date)`+`astype(float)`(WP5�Ѱ��˴���)
- `industry_map.parquet`: PIT��ҵӳ��(code,date->industry), merge_asof��ʷ����+��ǰ����, UNKNOWNռ��5.7%
