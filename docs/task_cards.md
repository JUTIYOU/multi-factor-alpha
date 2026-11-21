# P0 任务卡 v1.1(修正版,已并入接口核实结果)

> 完整12张任务卡见会话第一轮拆解;此处为落盘版,含接口核实修正。

## T1.0(并入T1.1) 交易日历+指数基准
- `query_trade_dates` → `data/raw/trade_calendar.parquet`
- `sh.000905` 指数日线(与个股同接口) → `data/raw/index_quotes.parquet`
- 用途:覆盖率验收、embargo/purge窗口、标签第h个交易日对齐、WP6/7基准

## T1.1 行情+复权因子拉取
- 字段: `date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST`
- adjustflag=3(不复权);复权另拉 `query_adjust_factor`(foreAdjustFactor/backAdjustFactor,本地算复权价)
- per-code: `data/raw/quotes/{code}.parquet` + `_checkpoint.json`;失败清单重试≤3
- 验收:覆盖率≥99%交易日(对trade_calendar);volume=股/amount=元断言

## T1.2 财务拉取+PIT对齐
- 接口×4: profit(roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare)、
  growth(YOYSales,YOYProfitBasic等同比字段)、balance(liabilityToAsset等)、cash_flow(NCFOperateA等,累计口径)
- 按(code,year,quarter)拉取,2017Q1起;`available_date=pubDate`,缺失→statDate+90d+imputed标记
- per-code: `data/raw/financials/{interface}/{code}.parquet`
- 验收:抽样断言max(available_date)≤date;流量字段累计口径已登记(差分在因子层处理)

## T1.3 成分快照+行业+上市信息
- `query_zz500_stocks(date=月末交易日)` → 历史成分快照(PIT真实)
- `query_stock_basic` → ipoDate/outDate/type/status
- `query_stock_industry` → 行业(当前口径,披露)
- 产出:`data/raw/zz500_snapshots.parquet`,`stock_basic.parquet`,`industry.parquet`

## T1.4 可交易掩膜
`tradable = in_universe(t) & (tradestatus==1) & (isST==0) & (list_days>=60)`
- in_universe用月末快照插值到日;list_days用ipoDate

## T1.5 清洗面板+数据字典
- panel合并:quotes + badj价 + 掩膜 + 市值(close×pubDate门控totalShare)
- (code,date)唯一断言;单位断言;缺失率按字段×年份
- `data/processed/panel.parquet`, `reports/data_dictionary.md`

## T1.6 ★硬闸门 质量报告+PIT抽查+对拍
- PIT抽查20例(code,date):as-of join后max(available_date)≤date断言
- 估值对拍20例:自建pubDate门控BP vs 1/pbMRQ,偏离>10%→弃用接口估值
- 内部一致性:复权因子跳变检查、pctChg vs 后复权差分对拍、停牌日字段形态
- `reports/data_quality.md` + `pytest tests/test_pit.py`


## ���״̬ 2026-11-16
- T0-T1.6 ������·ȫ��: GATE_T16.ok
- WP1-WP5 ȫ������: reports/wp{1..5}_acceptance.md
- WP6 �չ�: final_report.md + resume_bullets.md + README����·��
- δ��(��ѡ��): �ı���������������(���ݿɿ��Բ���,�������ų�)
