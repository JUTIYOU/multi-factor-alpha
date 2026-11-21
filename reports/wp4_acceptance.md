# WP4 验收报告 — SHAP归因审计 + 稳定性监控 + IC衰减

日期: 2026-11-16 | 结论: **通过** (复现验证 corr=1.000000, 输出独立复核一致)

## 1. SHAP归因 (P3协议各折LGBM重训, test折tradable样本TreeSHAP)

**簇级重要性份额** (|ρ|≥0.7聚类防信用稀释):

| 簇 | h5 share | h10 share | 成员 |
|---|---|---|---|
| 流动性簇 | 16.5% | 21.1% | turn_mean, amount_mean |
| 日内行为簇 | 12.4% | 13.3% | intraday |
| 波动簇 | 11.1% | 14.3% | ivol, amplitude, down_vol等 |
| 基本面 | ~18%合计 | ~16% | g_np, eps_ttm, roe 各自独立贡献 |

- **折间重要性排名相关 h5=0.768 / h10=0.830** —— 重要性结构跨时间稳定,非随机拟合噪声
- 高SHAP因子的独立性检验: 流动性/波动簇经indsize中性化后仍高贡献 = **独立alpha**; 基本面因子(roe/g_np)贡献稳定但排名居中
- 复现性: fold0重训预测 vs 落盘pred **corr=1.000000** ✓
- 筛选稳定性: EP/amount_mean/ivol/intraday/vol_ratio/vol_asym/pv_corr/turn_mean/turn_chg **全部27折入选**; rev_5仅1折(模型偏好rev_20, 23折)

## 2. 漂移归因 (factor_drift.csv, 回答"哪个因子导致漂移")

| 因子 | psi_max | 报警季 | ic_decay_ratio | 解读 |
|---|---|---|---|---|
| lev | 4.95 | 6 | -0.52 | 分布剧烈漂移+IC翻负, 但未入选白名单 |
| mom_12_1 | 0.79 | 8 | 1.07 | 分布漂移但预测力稳定 |
| roe | 0.49 | 7 | **0.42** | 真实alpha衰减,需监控 |
| npm | 0.35 | 2 | **0.26** | 真实alpha衰减 |
| ivol/vol_60/amplitude | 0.3-0.47 | 3-5 | 1.2-1.3 | 分布漂移但IC反而增强 → 慢变量拥挤度变化,非alpha消失 |
| EP | 0.34 | 2 | 0.92 | 稳定 |

**漂移机制区分**: 分布漂移(PSI)与预测力衰减(IC decay)不同步——波动类分布漂了但alpha还在; 基本面类是真衰减。

## 3. IC半衰期 (h=1..20)

- **绝大多数因子 halflife>20**: 流动性/波动类 |IC| 随h**单调增强**(ivol -0.032→-0.063, turn_mean -0.027→-0.057) → 慢变量信号, 更长持有期捕捉更多alpha → 支撑h=10主线
- rev_5 平滑衰减 h1=0.034→h20=0.018(保留53%) → 快反转因子但仍持久
- max_dd/g_rev halflife=2 → 快衰减, 与初筛淘汰一致(交叉验证✓)

## 4. 合成信号稳定性 (composite_drift.csv)

| h | proto | ic_mean | roll63_min | roll63_last | 负IC日占比 |
|---|---|---|---|---|---|
| 5 | P3/P4 | .053/.055 | 0.001/-0.004 | .066/.067 | ~30% |
| 10 | P3/P4 | .055/.059 | -0.050/-0.022 | .085/.091 | ~30% |

滚动IC曾触底~0/-0.05(对应2024小微盘崩盘等regime切换期), 但均恢复 → 无永久性失效。

## 5. 修复记录(诚实)

- exposure_persistence: pivot_table dropna丢全NaN日期行 → reindex对齐修复
- composite_drift: 变量复用致h5/P4漏算 → 补算落盘
- drift_summary: index name丢失致列名错误 → 修复
- ic_halflife: 哨兵值20歧义 → 改">20"字符串标记

## 6. 产出文件

`reports/fold_selection.csv` `shap_importance.csv` `factor_drift.csv` `ic_halflife.csv` `composite_drift.csv`

## 7. 门禁裁决

SHAP审计+漂移归因完成 → **WP5可启动**: 含成本分层回测 + 组合风险归因(行业/风格/残差)。
