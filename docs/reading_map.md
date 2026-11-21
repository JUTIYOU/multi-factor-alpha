# 阅读地图 — 必须掌握的核心代码

按依赖顺序读，每文件标注"必须理解的点"：

| 顺序 | 文件 | 必须掌握的点 |
|---|---|---|
| 1 | `src/data/pit.py` | `available_date=pubDate` 语义；`merge_asof(direction="backward")` 为什么防前视；断言双保险 |
| 2 | `src/data/panel.py` | **`add_badj`**：BaoStock `backAdjustFactor` 是**累计**因子（大坑，泄漏清单#19）；tradable 掩膜四条件；市值=close×pubDate门控股本 |
| 3 | `src/factors/pricevol.py` / `fundamental.py` | 滚动窗含停牌日口径 + 护栏；同比因子的双重 PIT 校验 |
| 4 | `src/factors/preprocess.py` + `neutralize.py` | clip→WLS残差→zscore 顺序为什么固定；WLS `w=sqrt(mktcap)`；拟合集=tradable、残差外推全体 |
| 5 | `src/factors/ic.py` | `shift(-h)` 前向收益口径；逐日 spearman；ICIR 年化 `×sqrt(252/h)` |
| 6 | `src/models/labels.py` + `splits.py` | rank 标签为什么只在 `groupby(date)` 内；**purge/embargo 的标签窗重叠原理**（泄漏清单核心） |
| 7 | `src/models/train.py` | `daily_ic_matrix`：pct-rank+corrwith=spearman 的向量化；**嵌套筛选防 selection bias**；train 尾 20% early-stop |
| 8 | `src/models/shap_audit.py` + `stability.py` | 簇级聚合防相关因子信用稀释；PSI 边界固定自训练窗 |
| 9 | `src/backtest/portfolio.py` | t+1 open 成交防"看收盘才能成交"的前视；一字板买卖规则 |
| 10 | `src/backtest/engine.py` | **固定权重 Σr/N**（大坑，泄漏清单#20）；边界日旧 exit+新 entry 同日记账 |
| 11 | `src/backtest/attribution.py` | 行业哑变量含市场水平→减均值=行业超额；四段分解恒等式 |

## 学习建议

- 先看 `AGENTS.md` 的硬约束再看代码——每个"为什么"都对应清单里的一条泄漏防线
- 两个大坑（复权因子、回测权重）在 `docs/leakage_checklist.md` #19/#20 有完整事故记录
- 跑 `tests/` 的合成测试是最快验证理解的方式：每个测试对应一个防呆不变量
