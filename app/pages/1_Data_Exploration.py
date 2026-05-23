"""Page 2 — Data Exploration. Tight grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    CLUSTER_NAMES,
    COLOR_MAP_BY_NAME,
    PALETTE,
    inject_css,
    mini_kpi,
    load_clustered_data,
    render_sidebar,
    style_fig,
    pill,
)

st.set_page_config(
    page_title="Exploration | Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

df = load_clustered_data()

st.sidebar.markdown("### 🎛️ Filters")
age_range = st.sidebar.slider(
    "Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max())),
)
income_range = st.sidebar.slider(
    "Income",
    int(df["Income"].min()),
    int(df["Income"].max()),
    (int(df["Income"].min()), int(df["Income"].max())),
    step=1000,
)
kids_opts = sorted(df["Kids"].unique().tolist())
kids = st.sidebar.multiselect("Kids", kids_opts, default=kids_opts)

filt = df[
    (df["Age"].between(*age_range))
    & (df["Income"].between(*income_range))
    & (df["Kids"].isin(kids))
].copy()
filt["Segment"] = filt["Cluster"].map(CLUSTER_NAMES)

hcol, k1, k2, k3, k4, k5 = st.columns([2.6, 1, 1, 1, 1, 1])
with hcol:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "📊 Data Exploration</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "Interactive EDA · sidebar filters update all charts</div>",
        unsafe_allow_html=True,
    )
with k1:
    mini_kpi(
        "Filtered",
        f"{len(filt):,}",
        f"of {len(df):,}",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k2:
    mini_kpi("Med income", f"${filt['Income'].median():,.0f}", "")
with k3:
    mini_kpi("Med spend", f"${filt['Expenses'].median():,.0f}", "")
with k4:
    mini_kpi("Avg purch", f"{filt['TotalNumPurchases'].mean():.1f}", "per cust")
with k5:
    mini_kpi("Avg age", f"{filt['Age'].mean():.0f}", "yrs")

# ROW 1
r1c1, r1c2 = st.columns([1.5, 1])

with r1c1:
    fig = px.scatter(
        filt,
        x="Income",
        y="Expenses",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        opacity=0.6,
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#1F2937",
    )
    fig.update_traces(marker=dict(size=5))
    style_fig(fig, height=300, show_legend=True, title="Income × Expenses (ρ≈0.82)")
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig = px.histogram(
        filt,
        x="Income",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        nbins=30,
        opacity=0.75,
    )
    fig.update_layout(barmode="overlay")
    style_fig(fig, height=145, title="Income distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(
        filt,
        x="Expenses",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        nbins=30,
        opacity=0.75,
    )
    fig.update_layout(barmode="overlay")
    style_fig(fig, height=145, title="Spend distribution")
    st.plotly_chart(fig, use_container_width=True)

# ROW 2: 4 small charts
r2 = st.columns(4)

with r2[0]:
    fig = px.scatter(
        filt,
        x="Age",
        y="TotalNumPurchases",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=4))
    style_fig(fig, height=180, title="Age × Purchases")
    st.plotly_chart(fig, use_container_width=True)

with r2[1]:
    kids_seg = filt.groupby(["Kids", "Segment"]).size().reset_index(name="n")
    fig = px.bar(
        kids_seg,
        x="Kids",
        y="n",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        barmode="stack",
    )
    style_fig(fig, height=180, title="Family size × Segment")
    st.plotly_chart(fig, use_container_width=True)

with r2[2]:
    fig = px.box(
        filt,
        x="Segment",
        y="Days_is_client",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        points=False,
    )
    style_fig(fig, height=180, title="Tenure (days)")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

with r2[3]:
    fig = px.box(
        filt,
        x="Segment",
        y="Recency",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        points=False,
    )
    style_fig(fig, height=180, title="Recency")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

# ROW 3: 4 correlation cards
corr = [
    ("Income ↔ Expenses", "0.82", "linear-gradient(135deg, #10B981 0%, #14B8A6 100%)"),
    ("Income ↔ Purchases", "0.71", "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"),
    ("Expenses ↔ Purch.", "0.76", "linear-gradient(135deg, #F59E0B 0%, #F97316 100%)"),
    (
        "Recency ↔ Response",
        "−0.20",
        "linear-gradient(135deg, #EC4899 0%, #EF4444 100%)",
    ),
]
r3 = st.columns(4)
for col, (label, val, grad) in zip(r3, corr):
    with col:
        st.markdown(
            f'<div style="background:{grad}; color:white; border-radius:10px; padding:10px 14px;">'
            f'<div style="font-size:10px; opacity:0.9; font-weight:600;">{label}</div>'
            f'<div style="font-size:24px; font-weight:800; margin-top:2px;">{val}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
