# -*- coding: utf-8 -*-
"""Streamlit dashboard: 多因子合成与风险归因 项目展示.

数据边界: 只读 reports/*.csv + reports/demo/index_nav.csv(部署小包),
不依赖 data/processed(不上传). 本地: streamlit run app.py
部署: Streamlit Community Cloud -> 入口 app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
REP = ROOT / "reports"

st.set_page_config(page_title="多因子合成与风险归因", layout="wide",
                   initial_sidebar_state="expanded")

# ---- 统一视觉 ----
C = dict(navy="#1f4e79", blue="#2e86de", lblue="#93c5fd", gray="#6b7280",
         teal="#0d9488", amber="#d97706", red="#dc2626", green="#15803d")
PALETTE = [C["blue"], C["amber"], C["teal"], C["red"], C["gray"],
           C["navy"], C["lblue"], C["green"]]

# 因子中文注解(与src/factors定义一致)
FACTOR_CN = {
    "rev_5": "5日反转(买跌卖涨)", "rev_10": "10日反转", "rev_20": "20日反转",
    "mom_12_1": "12-1月动量(剔除近月)",
    "vol_20": "20日收益波动率", "vol_60": "60日收益波动率",
    "ivol": "60日特异波动率(残差std)", "skew_20": "20日收益偏度",
    "down_vol": "20日下行波动率", "max_dd": "60日最大回撤",
    "turn_mean": "20日平均换手率", "turn_chg": "换手率变化(5日/60日)",
    "amount_mean": "20日平均成交额", "amplitude": "20日平均日内振幅",
    "vol_ratio": "量比(5日/60日成交量)", "pv_corr": "20日量价相关性",
    "vol_asym": "60日涨跌日量能不对称", "overnight": "20日平均隔夜跳空",
    "intraday": "20日平均日内收益",
    "BP": "账面市值比(1/PB)", "EP": "盈利收益率(1/PE)",
    "SP": "营收市值比(1/PS)", "CFP": "现金收益率(1/PCF)",
    "roe": "净资产收益率", "gpm": "毛利率", "npm": "净利率",
    "lev": "资产负债率", "g_rev": "营收同比增速", "g_np": "净利润同比增速",
    "cfo_np": "经营现金流/净利润", "eps_ttm": "每股收益TTM",
}


def flab(f: str) -> str:
    """因子下拉标签: code · 中文解释."""
    return f"{f} · {FACTOR_CN.get(f, '')}"


def style(fig: go.Figure, height: int = 420, title: str = "",
          legend_y: float = -0.16) -> go.Figure:
    """统一视觉; 图例放图下方(避免与标题/文字叠加),
    带x轴标题的图传legend_y=-0.28再下移."""
    fig.update_layout(
        template="plotly_white", height=height,
        title=dict(text=title, font=dict(size=15, color=C["navy"]),
                   x=0, xanchor="left"),
        font=dict(size=11, color="#374151"),
        margin=dict(l=46, r=20, t=52, b=78),
        legend=dict(orientation="h", yanchor="top", y=legend_y, x=0,
                    font=dict(size=10)),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="#eef2f7")
    fig.update_yaxes(gridcolor="#eef2f7")
    return fig


@st.cache_data
def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(REP / f"{name}.csv")
    for c in ("date", "reb"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


# ================= 头部 =================
S = load("backtest_summary")
row_main = S[(S.protocol == "P4") & (S.horizon == 10)].iloc[0]

st.markdown(
    f"<h2 style='color:{C['navy']};margin-bottom:0'>"
    f"多因子合成与风险归因 · 研究结果交互展示</h2>"
    f"<p style='color:{C['gray']}'>中证500股票池 · 日频 · 2018-2025 · "
    f"PIT防泄漏 · 行业+市值中性化 · 含成本回测</p>",
    unsafe_allow_html=True)

m = st.columns(6)
m[0].metric("净年化收益(P4 h10)", f"{row_main.ann_ret:.1%}")
m[1].metric("夏普比率", f"{row_main.sharpe:.2f}")
m[2].metric("最大回撤", f"{row_main.maxdd:.1%}")
m[3].metric("中证500同期", f"{row_main.bench_total:.1%}")
m[4].metric("风格伪装占比", "19–26%")
m[5].metric("残差alpha占比", "4–16%")
st.divider()

# ================= 侧栏 =================
PROTO_LAB = {"P4": "P4 · 等权合成(主线)", "P3": "P3 · LightGBM合成"}
VAR_LAB = {"indsize": "行业+市值中性化", "size": "仅市值中性化",
           "raw": "原始(未中性化)"}

with st.sidebar:
    st.markdown("### 分析口径")
    proto = st.radio("信号协议", ["P4", "P3"],
                     format_func=lambda p: PROTO_LAB[p])
    h = st.radio("持有周期", [10, 5], horizontal=True,
                 format_func=lambda x: f"{x} 个交易日",
                 help="信号持有h日后调仓; h10换手更低成本更少")
    variant = st.radio("特征版本", ["indsize", "size", "raw"],
                       format_func=lambda v: VAR_LAB[v],
                       help="影响[协议与合成]页; 因子解剖页本身就在对比三版本")
    st.markdown("### 显示选项")
    show_gross = st.checkbox("对比毛净值(不含成本)", value=True)
    show_bench = st.checkbox("对比中证500", value=True)
    with st.expander("名词解释"):
        st.markdown("""
- **P1** 一次性训练永不重训 · **P2** 走样季重训 ·
  **P3** P2+purge/embargo防泄漏(模型主线) · **P4** 等权合成(信号主线)
- **raw** 因子原值 · **size** 剥市值 · **indsize** 剥行业+市值
- **IC** 因子排名与未来收益排名的日度相关 · **ICIR** IC均值/IC波动
- **PSI** 分布漂移度(>0.25报警) · **IC衰减比** 后期IC/前期IC(<1=衰减)
- **残差alpha** 组合收益中市场/行业/市值解释不了的部分
""")
    st.divider()
    st.caption("数据源: BaoStock日频 · 中证500历史成分\n"
               "所有结果均为样本外/含成本口径")

t0, t1, t2, t3, t4, t5, t6 = st.tabs(
    ["数据与防线", "回测总览", "因子解剖", "协议与合成", "SHAP审计",
     "漂移监控", "归因与关于"])

# ================= Tab0 数据与防线 =================
with t0:
    st.markdown("**数据规模与质量门禁** —— 一切结论建立在这些数字上")
    c0 = st.columns(6)
    for c_, (lab, val) in zip(c0, [
            ("面板行数", "241万"), ("覆盖股票", "1217只"),
            ("交易日", "2186"), ("覆盖率", "100%"),
            ("可交易占比", "44.3%"), ("PIT抽查", "20/20")]):
        c_.metric(lab, val)

    tc = pd.read_csv(REP / "demo" / "tradable_daily.csv",
                     parse_dates=["date"])
    figt = go.Figure(go.Scatter(
        x=tc.date, y=tc.n_tradable, mode="lines",
        line=dict(color=C["blue"], width=1.2),
        fill="tozeroy", fillcolor="rgba(46,134,222,0.15)"))
    figt.update_yaxes(title="每日可交易股票数")
    st.plotly_chart(style(figt, 300, "可交易池规模(2017-2025, 451-500只/日)"),
                    width="stretch")

    da, db = st.columns(2)
    with da:
        st.markdown("**防泄漏防线(泄漏清单20条)**")
        st.markdown("""
- **PIT单入口**: 财务/股本可用性只在`pit.py`判断
- **时间连接**: 一律`merge_asof(backward)`+断言兜底
- **截面纪律**: 一切统计量只在`groupby('date')`内
- **嵌套筛选**: 每折只用训练窗ICIR选因子
- **切边卫生**: purge+embargo砍标签窗重叠样本
- **执行诚实**: t收盘信号→t+1开盘成交,一字板处理
- **记账口径**: 组合日收益=固定权重Σr/N
""")
    with db:
        st.markdown("**重大bug: 均已修复并有回归测试**")
        st.success("✅ **#19 复权因子重复cumprod** —— `backAdjustFactor`"
                   "本身已是累计因子,再乘一次致远期收益爆炸(+2400万%)。\n\n"
                   "**修复**: as-of取最近一次事件的累计因子,"
                   "首事件前用`BF/f_true`反推基线。\n\n"
                   "**验证**: fwd5最大值+2400万%→+48.8%,"
                   "badj差分异常8206→12行; `test_badj.py`回归测试。\n\n"
                   "教训:第三方字段语义必须小样本实证")
        st.success("✅ **#20 在场腿均值记账** —— 顺延腿只剩1-3只时"
                   "个股日行被放大成全组合收益,7年虚增+157%。\n\n"
                   "**修复**: 组合日收益改固定权重Σr/N。\n\n"
                   "**验证**: gross 4.31→1.92,与独立腿口径对齐(差1.3%已披露);"
                   "`test_fixed_weight_not_present_mean`回归测试。\n\n"
                   "教训:权重只能来自构建约定,不能来自数据可得性")
        st.caption("完整事故记录见`docs/leakage_checklist.md` #19/#20,"
                   "测试17/17通过。")

# ================= Tab1 回测总览 =================
with t1:
    nav = load("backtest_nav")
    layers = load("backtest_layers")
    bench = pd.read_csv(REP / "demo" / "index_nav.csv",
                        parse_dates=["date"])
    d = nav[(nav.protocol == proto) & (nav.horizon == h)]
    r = S[(S.protocol == proto) & (S.horizon == h)].iloc[0]

    cols = st.columns(6)
    items = [("净年化", f"{r.ann_ret:.1%}"), ("毛年化", f"{r.gross_ann:.1%}"),
             ("夏普", f"{r.sharpe:.2f}"), ("最大回撤", f"{r.maxdd:.1%}"),
             ("年换手", f"{r.turnover_ann:.1f}x"),
             ("成本拖累/年", f"{r.cost_drag_ann:.1%}")]
    for c_, (lab, val) in zip(cols, items):
        c_.metric(lab, val)

    dd = d.nav_net / d.nav_net.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_scatter(x=d.date, y=d.nav_net, name="净净值(含成本)",
                    line=dict(color=C["blue"], width=1.8), row=1, col=1)
    if show_gross:
        fig.add_scatter(x=d.date, y=d.nav_gross, name="毛净值(不含成本)",
                        line=dict(color=C["lblue"], width=1.2, dash="dash"),
                        row=1, col=1)
    if show_bench:
        fig.add_scatter(x=bench.date, y=bench.bench_nav, name="中证500",
                        line=dict(color=C["gray"], width=1.1), row=1, col=1)
    fig.add_scatter(x=d.date, y=dd * 100, name="回撤%",
                    fill="tozeroy", fillcolor="rgba(220,38,38,0.25)",
                    line=dict(color=C["red"], width=0.8), row=2, col=1)
    st.plotly_chart(style(fig, 520,
                          f"{proto} h{h} — 多头组(前20%)净值 vs 中证500"),
                    width="stretch")
    st.caption(f"执行: t收盘信号 → t+1开盘买 → t+1+h开盘卖 · "
               f"买5bp/卖15bp · 一字板跳过率 {r.skip_rate:.2%}")

    st.markdown("**五分位分层累计收益** —— 组合级单调性(Q5=多头组)")
    lay = layers[(layers.protocol == proto) & (layers.horizon == h)]
    figl = go.Figure()
    colors = {1: C["lblue"], 2: "#a5b4fc", 3: C["gray"],
              4: C["amber"], 5: C["blue"]}
    for g in range(1, 6):
        figl.add_scatter(x=lay.reb, y=lay[str(g)], name=f"Q{g}",
                         line=dict(color=colors[g],
                                   width=1.8 if g == 5 else 1.0))
    st.plotly_chart(style(figl, 380), width="stretch")

# ================= Tab2 因子解剖 =================
with t2:
    ic = load("ic_halflife")
    neu = load("factor_neutralization")
    scr = load("factor_screening")
    factors = sorted(ic.factor.unique())

    st.markdown("**IC衰减曲线** —— 慢信号(|IC|随h增强) vs 快信号")
    picks = st.multiselect("选择因子", factors, format_func=flab,
                           default=[f for f in
                                    ["rev_5", "ivol", "turn_mean", "roe"]
                                    if f in factors])
    fig = go.Figure()
    for f in picks:
        s = ic[ic.factor == f].sort_values("h")
        fig.add_scatter(x=s.h, y=s.ic, name=f, mode="lines+markers")
    fig.add_hline(y=0, line_color="#9ca3af", line_width=0.8)
    fig.update_xaxes(title="持有期 h(交易日)")
    fig.update_yaxes(title="逐日RankIC均值")
    st.plotly_chart(style(fig, 380, legend_y=-0.28), width="stretch")

    c2a, c2b = st.columns([1, 1])
    with c2a:
        st.markdown("**单调性审计** —— top15/26, "
                    "十分位收益与因子排名的Spearman")
        s5 = (scr[(scr.horizon == h) & scr.selected]
              .assign(absrho=lambda x: x.mono_rho.abs())
              .sort_values("absrho", ascending=False).head(15)
              .sort_values("mono_rho"))
        figm = go.Figure(go.Bar(
            x=s5.mono_rho, y=s5.factor, orientation="h",
            marker_color=[C["blue"] if v > 0 else C["red"]
                          for v in s5.mono_rho],
            text=s5.mono_rho.round(2), textposition="auto"))
        figm.update_xaxes(title="十分位收益 × 因子排名 Spearman")
        st.plotly_chart(style(figm, 460, legend_y=-0.28),
                        width="stretch")
    with c2b:
        st.markdown("**中性化代价** —— 三版对比")
        f1 = st.selectbox("因子", factors, format_func=flab,
                          index=factors.index("mom_12_1")
                          if "mom_12_1" in factors else 0)
        n5 = neu[(neu.factor == f1) & (neu.horizon == h)]
        for metric, ttl in [("ic_mean", "|日均IC|"), ("icir", "|ICIR|")]:
            figb = go.Figure(go.Bar(
                x=n5.variant, y=n5[metric].abs(),
                marker_color=[C["gray"], C["amber"], C["blue"]],
                text=n5[metric].round(4), textposition="auto"))
            st.plotly_chart(style(figb, 210, f"{f1} {ttl}"),
                            width="stretch")

    with st.expander("初筛全表 (26/31入选)"):
        show = scr[scr.horizon == h].drop(
            columns=["variant", "horizon"], errors="ignore")
        st.dataframe(show.sort_values("icir", ascending=False),
                     hide_index=True, width="stretch")

# ================= Tab3 协议与合成 =================
with t3:
    pc = load("model_protocol_comparison")
    comp = load("composite_drift")
    sel = load("fold_selection")

    st.markdown("**四协议对照** —— 同一测试区间, 差值干净归因 "
                f"(当前: {VAR_LAB[variant]})")
    sub = (pc[(pc.variant == variant) & (pc.horizon == h)]
           .set_index("protocol").reindex(["P1", "P2", "P3", "P4"])
           .reset_index())
    if sub.ic_mean.notna().sum() == 0:
        st.info("该特征版本未跑协议对比——消融实验只需两个端点"
                "(原始 vs 行业+市值中性化)。下方展示单因子层面"
                "的三版对比，size 介于两端之间。")
        nn = load("factor_neutralization")
        nn = nn[nn.horizon == h]
        agg = (nn.groupby("variant")[["ic_mean", "icir"]]
               .apply(lambda x: x.abs().mean()).reindex(
                   ["raw", "size", "indsize"]))
        st.markdown("**单因子层面三版对比(全体因子平均)** —— "
                    "size 落在 raw 与 indsize 之间")
        na, nb = st.columns(2)
        for col_, metric, ttl in [(na, "ic_mean", "平均|日均IC|"),
                                  (nb, "icir", "平均|ICIR|")]:
            fign = go.Figure(go.Bar(
                x=agg.index, y=agg[metric],
                marker_color=[C["gray"], C["amber"], C["blue"]],
                text=agg[metric].round(3), textposition="auto"))
            col_.plotly_chart(style(fign, 300, ttl), width="stretch")
    else:
        a, b = st.columns(2)
        for col_, metric, ttl, fmt in [(a, "ic_mean", "日均IC", ".4f"),
                                       (b, "icir", "ICIR", ".3f")]:
            figp = go.Figure(go.Bar(
                x=sub.protocol, y=sub[metric],
                marker_color=[C["gray"], C["teal"], C["blue"], C["amber"]],
                text=[f"{v:{fmt}}" for v in sub[metric]],
                textposition="auto"))
            col_.plotly_chart(
                style(figp, 320, f"{VAR_LAB[variant]} h{h} — {ttl}"),
                width="stretch")
    st.caption("P1 训一次·P2 走样重训·P3 +purge/embargo·P4 等权合成 —— "
               "P1→P2=重训收益(+20%IC), P2→P3=切边水分(≈0), P3→P4=非线性增量(≈0或负)")

    c3a, c3b = st.columns([1, 1])
    with c3a:
        st.markdown("**每折入选因子频次** —— 嵌套筛选的稳定性")
        fs = sel[sel.horizon == h]
        freq = (fs.groupby("factor").size()
                .sort_values().rename("n_folds").reset_index())
        figf = go.Figure(go.Bar(
            x=freq.n_folds, y=freq.factor, orientation="h",
            marker_color=C["teal"], text=freq.n_folds,
            textposition="auto"))
        figf.update_xaxes(title=f"入选折数 (共{fs.fold.max()+1}折)")
        st.plotly_chart(style(figf, 440, legend_y=-0.28),
                        width="stretch")
    with c3b:
        st.markdown("**合成信号稳定性** —— 滚动63日IC")
        ric = pd.read_csv(REP / "demo" / "roll_ic.csv",
                          parse_dates=["date"])
        cur = ric[(ric.protocol == proto) & (ric.horizon == h)]
        comp_h = comp[(comp.horizon == h) &
                      (comp.protocol == proto)].iloc[0]
        figr = go.Figure()
        for (p_, h_), gg in ric.groupby(["protocol", "horizon"]):
            is_cur = (p_ == proto and h_ == h)
            figr.add_scatter(x=gg.date, y=gg.roll_ic,
                             name=f"{p_} h{h_}",
                             line=dict(width=2.0 if is_cur else 0.9,
                                       color=C["blue"] if is_cur
                                       else "#cbd5e1"),
                             opacity=1.0 if is_cur else 0.7)
        figr.add_hline(y=0, line_color=C["red"], line_dash="dash")
        st.plotly_chart(
            style(figr, 300,
                  f"滚动63日IC — {proto} h{h} 粗线(其余臂淡显)"),
            width="stretch")
        cc = st.columns(3)
        cc[0].metric("滚动IC最低", f"{comp_h.roll_min:.3f}")
        cc[1].metric("滚动IC最高", f"{comp_h.roll_max:.3f}")
        cc[2].metric("负IC日占比", f"{comp_h.neg_days_pct:.0%}")
        st.caption("2024微盘崩盘期滚动IC曾触底≈0/-0.05(图中可见凹陷), "
                   "随后恢复——信号短期失效但结构未破。")

# ================= Tab4 SHAP审计 =================
with t4:
    shp = load("shap_importance")
    s4 = shp[shp.horizon == h]
    st.markdown("**簇级份额** —— 先按|ρ|>0.7聚类再聚合, 防相关因子信用稀释")
    cl = (s4.drop_duplicates("cluster")[["cluster", "cluster_share"]]
            .sort_values("cluster_share"))
    figc = go.Figure(go.Bar(
        x=cl.cluster_share * 100, y=cl.cluster, orientation="h",
        marker_color=C["navy"], text=(cl.cluster_share * 100).round(1),
        textposition="auto"))
    figc.update_xaxes(title="占全部|SHAP|份额 %")
    st.plotly_chart(style(figc, 400, f"簇级重要性份额, h{h}(样本外)",
                          legend_y=-0.28), width="stretch")

    st.markdown("**单因子排名** —— 文本为折内平均名次, 误差条为名次波动")
    top = s4.sort_values("mean_abs_shap", ascending=False).head(15)
    figf2 = go.Figure(go.Bar(
        x=top.mean_abs_shap, y=top.factor, orientation="h",
        marker_color=C["blue"], text=top.mean_rank.round(1),
        textposition="auto"))
    figf2.update_yaxes(autorange="reversed")
    figf2.update_xaxes(title="mean |SHAP|")
    st.plotly_chart(style(figf2, 460, legend_y=-0.28), width="stretch")
    st.caption("折间重要性排名相关 0.77–0.83 —— 重要性结构跨时间稳定;"
               "高贡献因子为独立alpha而非风格代理。")

    with st.expander("相关簇成员名单(|ρ|>0.7聚类, 簇内选代表)"):
        memb = (s4.groupby("cluster")["factor"]
                .apply(lambda x: "、".join(sorted(x)))
                .rename("簇内因子").reset_index())
        st.dataframe(memb, hide_index=True, width="stretch")

# ================= Tab5 漂移监控 =================
with t5:
    dr = load("factor_drift")
    thr = st.slider("PSI报警阈值", 0.1, 1.0, 0.25, 0.05)
    st.markdown("**漂移 vs 衰减** —— 两个独立的失效维度")
    alarmed = dr.psi_max > thr
    figd = go.Figure()
    figd.add_scatter(
        x=dr.loc[~alarmed, "psi_max"],
        y=dr.loc[~alarmed, "ic_decay_ratio"],
        mode="markers", name="正常",
        marker=dict(size=10, color=C["blue"], opacity=0.7),
        text=dr.loc[~alarmed, "factor"],
        hovertemplate="%{text}<br>PSI %{x:.2f} · 衰减比 %{y:.2f}<extra></extra>")
    figd.add_scatter(
        x=dr.loc[alarmed, "psi_max"], y=dr.loc[alarmed, "ic_decay_ratio"],
        mode="markers", name="PSI报警",
        marker=dict(size=11, color=C["red"], opacity=0.85,
                    symbol="diamond"),
        text=dr.loc[alarmed, "factor"],
        hovertemplate="%{text}<br>PSI %{x:.2f} · 衰减比 %{y:.2f}<extra></extra>")
    figd.add_vline(x=thr, line_color=C["red"], line_dash="dash")
    figd.add_hline(y=1.0, line_color=C["gray"], line_dash="dash")
    figd.update_xaxes(title="最大PSI(分布漂移)")
    figd.update_yaxes(title="IC后期/前期比(预测力衰减)")
    st.plotly_chart(style(figd, 470, legend_y=-0.28), width="stretch")
    st.caption("右下象限(高PSI+低衰减比)=真alpha衰减, 如lev(0.42/-0.52类); "
               "高PSI但IC完好=分布漂移非失效, 如ivol。")

    st.markdown("**报警因子清单** —— 按报警季数与衰减比排序")
    alarm_tbl = (dr[dr.alarm_q > 0]
                 .sort_values(["alarm_q", "ic_decay_ratio"],
                              ascending=[False, True])
                 [["factor", "psi_max", "alarm_q", "ic_decay_ratio",
                   "persist_mean"]]
                 .rename(columns={
                     "factor": "因子", "psi_max": "最大PSI",
                     "alarm_q": "报警季数", "ic_decay_ratio": "IC衰减比",
                     "persist_mean": "暴露持续性"})
                 .round(3))
    st.dataframe(alarm_tbl, hide_index=True, width="stretch")

# ================= Tab6 归因+关于 =================
with t6:
    at = load("backtest_attribution")
    d6 = at[(at.protocol == proto) & (at.horizon == h)].sort_values("reb")

    st.markdown("**归因占比速览** —— 组合收益是谁赚来的")
    fin = d6.iloc[-1]
    tot = fin.cum_ret if fin.cum_ret != 0 else 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场β贡献", f"{fin.cum_market/tot:.0%}",
              f"累计 {fin.cum_market:+.1%}")
    c2.metric("行业倾斜贡献", f"{fin.cum_industry/tot:.0%}",
              f"累计 {fin.cum_industry:+.1%}")
    c3.metric("市值风格贡献", f"{fin.cum_size/tot:.0%}",
              f"累计 {fin.cum_size:+.1%}")
    c4.metric("残差alpha贡献", f"{fin.cum_residual/tot:.0%}",
              f"累计 {fin.cum_residual:+.1%}")
    st.caption("读法: 市场段是组合收益主体(常态), 残差段才是因子本身赚到的钱; "
               "size≈0 说明行业+市值中性化已传导到组合层——收益里没有大小盘风格水分。")

    st.markdown("**alpha 是稳定赚的还是靠运气** —— 每期再平衡残差贡献")
    pos = (d6.residual > 0).mean()
    figd = go.Figure(go.Bar(
        x=d6.reb, y=d6.residual * 100,
        marker_color=[C["blue"] if v > 0 else "#DC2626"
                      for v in d6.residual],
        hovertemplate="%{x|%Y-%m-%d}<br>残差贡献 %{y:+.2f}%<extra></extra>"))
    figd.add_hline(y=0, line_color="#94A3B8", line_width=1)
    figd.update_yaxes(title="残差贡献 %/期")
    st.plotly_chart(style(figd, 300,
                          f"{proto} h{h} 每期残差 (正占 {pos:.0%})"),
                    width="stretch")
    st.caption(f"正贡献期数占比 {pos:.0%}, 均值 {d6.residual.mean()*100:+.2f}%/期"
               f" —— 分散的小额正贡献比偶发大额更可信。")

    st.markdown("**四段累计分解** —— 贡献随时间的构成")
    figa = go.Figure()
    for col_, lab, colr in [("cum_market", "市场β", C["gray"]),
                            ("cum_industry", "行业倾斜", C["amber"]),
                            ("cum_size", "市值风格", C["teal"]),
                            ("cum_residual", "残差(alpha)", C["blue"])]:
        figa.add_scatter(x=d6.reb, y=d6[col_], name=lab,
                         stackgroup="one",
                         line=dict(color=colr, width=0.6), opacity=0.85)
    figa.add_scatter(x=d6.reb, y=d6.cum_ret, name="合计",
                     line=dict(color="#111827", width=1.5))
    figa.update_yaxes(title="累计贡献")
    st.plotly_chart(style(figa, 430, f"{proto} h{h} 累计分解"),
                    width="stretch")
    st.caption("看蓝色残差段的斜率: 平稳上行=alpha持续产出; 走平期=信号在该行情失灵。")

    st.markdown("**四臂终值分解对比** —— size≈0验证中性化传导到组合层")
    last = (at.sort_values("reb").groupby(["protocol", "horizon"])
              [["cum_market", "cum_industry", "cum_size", "cum_residual"]]
              .last() * 100)
    last.index = [f"{p} h{hh}" for p, hh in last.index]
    figb = go.Figure()
    for col_, lab, colr in [("cum_market", "市场β", C["gray"]),
                            ("cum_industry", "行业", C["amber"]),
                            ("cum_size", "市值", C["teal"]),
                            ("cum_residual", "残差", C["blue"])]:
        figb.add_bar(x=last.index, y=last[col_], name=lab,
                     marker_color=colr)
    figb.update_layout(barmode="stack")
    figb.update_yaxes(title="累计贡献 %")
    st.plotly_chart(style(figb, 380), width="stretch")
    best = last.cum_residual.idxmax()
    worst_sz = last.cum_size.abs().max()
    st.caption(f"残差最高: **{best}** (累计 {last.cum_residual.max():.1f}pct); "
               f"四臂市值贡献绝对值最大仅 {worst_sz:.1f}pct —— 风格水分已剥净, "
               f"各臂差异集中在残差段, 即因子筛选与合成方式本身。")

    with st.expander("关于本项目 — 方法/防线/诚实记录", expanded=False):
        st.markdown("""
**研究流水线**: BaoStock日频数据 → PIT门控面板(唯一入口`pit.py`,
`pubDate`可用日+asof-backward连接) → 31个截面因子 → 三版中性化
(raw/size/行业+市值WLS残差) → 四协议对照训练 → 折级SHAP簇级聚合 →
PSI+IC衰减漂移监控 → 次日开盘含成本回测 → 组合四段归因。

**归因方法**: 每期组合收益被恒等分解为四段——市场(基准收益) + 行业
(组合行业权重偏离基准的超配贡献) + 市值(市值暴露贡献) + 残差(剥离
以上风格后剩下的, 即因子本身赚到的钱)。市场段通常是主体(60-90%),
这符合任何多头组合的现实; 评估因子价值看残差段的持续性与占比。

**防泄漏防线**: PIT单入口 · merge_asof断言 · 截面groupby('date')纪律 ·
嵌套筛选 · purge+embargo · 次日开盘成交 · 固定权重记账。

**诚实记录**: 两个重大bug已写入`docs/leakage_checklist.md`——
#19累计复权因子重复cumprod(收益爆炸), #20在场腿均值代替固定权重
(净值虚增157%), 均已修复并有回归测试。

**诚实的负结果**: 正交因子上等权≥LGBM; 本场景切边泄漏≈0;
残差alpha占组合收益4-16%——其余是市场β与行业倾斜, 这是常态而非失败。
""")
