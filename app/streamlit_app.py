"""Page 1 — Overview dashboard. BI-style tight grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    CLUSTERS,
    CLUSTER_NAMES,
    COLOR_MAP_BY_NAME,
    PALETTE,
    inject_css,
    mini_kpi,
    load_clustered_data,
    load_profiles,
    load_tsne,
    render_sidebar,
    seg_chip,
    style_fig,
)

st.set_page_config(
    page_title="Overview | Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

df = load_clustered_data()
profiles = load_profiles()
tsne_df = load_tsne()

# HEADER + KPI STRIP (1 row)
header_col, k1, k2, k3, k4, k5 = st.columns([2.6, 1, 1, 1, 1, 1])

with header_col:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "🛍️ Customer Segmentation</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "K-Means · PCA · t-SNE · 2,198 retail customers</div>",
        unsafe_allow_html=True,
    )

total_customers = len(df)
total_revenue = df["Expenses"].sum()
n_champ = int((df["Cluster"] == 0).sum())
n_loyal = int((df["Cluster"] == 2).sum())
n_ps = int((df["Cluster"] == 1).sum())

with k1:
    mini_kpi(
        "Customers",
        f"{total_customers:,}",
        "cleaned",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k2:
    mini_kpi("Revenue", f"${total_revenue / 1000:,.0f}K", "total spend")
with k3:
    mini_kpi(
        "Champions",
        f"{n_champ}",
        f"{n_champ / total_customers * 100:.1f}%",
        gradient="linear-gradient(135deg, #F59E0B 0%, #F97316 100%)",
    )
with k4:
    mini_kpi(
        "Loyal VIPs",
        f"{n_loyal:,}",
        f"{n_loyal / total_customers * 100:.1f}%",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k5:
    mini_kpi(
        "Price-Sens.",
        f"{n_ps:,}",
        f"{n_ps / total_customers * 100:.1f}%",
        gradient="linear-gradient(135deg, #EF4444 0%, #EC4899 100%)",
    )

# ROW 2: t-SNE | 3 chips | donut
r1c1, r1c2, r1c3 = st.columns([1.5, 0.9, 1.1])

with r1c1:
    fig = px.scatter(
        tsne_df,
        x="tsne_x",
        y="tsne_y",
        color=tsne_df["Cluster"].map(CLUSTER_NAMES),
        color_discrete_map=COLOR_MAP_BY_NAME,
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    style_fig(fig, height=240, show_legend=True, title="t-SNE projection")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    for cid in [0, 2, 1]:
        seg_chip(cid, int((df["Cluster"] == cid).sum()), len(df))
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

with r1c3:
    rev_by = df.groupby("Cluster")["Expenses"].sum().reset_index()
    rev_by["Segment"] = rev_by["Cluster"].map(CLUSTER_NAMES)
    fig = px.pie(
        rev_by,
        values="Expenses",
        names="Segment",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        hole=0.6,
    )
    fig.update_traces(textinfo="percent", textfont=dict(size=10, color="white"))
    style_fig(fig, height=240, show_legend=True, title="Revenue share")
    fig.update_layout(
        annotations=[
            dict(
                text=f"${total_revenue / 1000:.0f}K",
                x=0.5,
                y=0.5,
                font=dict(size=14),
                showarrow=False,
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

# ROW 3: 4 small bar charts
r2 = st.columns(4)
prof = profiles.reset_index().copy()
prof["Segment"] = prof["Cluster"].map(CLUSTER_NAMES)

charts = [
    ("Income", "Income (median)", ",.0f", "$"),
    ("Expenses", "Spend (median)", ",.0f", "$"),
    ("TotalNumPurchases", "Purchases (median)", ",.0f", ""),
    ("TotalAcceptedCmp", "Campaigns accepted", ",.2f", ""),
]
for col, (metric, title, fmt, prefix) in zip(r2, charts):
    with col:
        fig = px.bar(
            prof,
            x="Segment",
            y=metric,
            color="Segment",
            color_discrete_map=COLOR_MAP_BY_NAME,
            text=prof[metric].apply(lambda v: f"{prefix}{format(v, fmt)}"),
        )
        fig.update_traces(textposition="outside", textfont=dict(size=9))
        style_fig(fig, height=180, title=title)
        fig.update_xaxes(tickfont=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

# ROW 4: distributions
df_plot = df.copy()
df_plot["Segment"] = df_plot["Cluster"].map(CLUSTER_NAMES)

r3 = st.columns(4)
with r3[0]:
    fig = px.histogram(
        df_plot,
        x="Income",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        nbins=30,
        opacity=0.75,
    )
    fig.update_layout(barmode="overlay")
    style_fig(fig, height=180, title="Income distribution")
    st.plotly_chart(fig, use_container_width=True)
with r3[1]:
    fig = px.histogram(
        df_plot,
        x="Expenses",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        nbins=30,
        opacity=0.75,
    )
    fig.update_layout(barmode="overlay")
    style_fig(fig, height=180, title="Spend distribution")
    st.plotly_chart(fig, use_container_width=True)
with r3[2]:
    fig = px.box(
        df_plot,
        x="Segment",
        y="Age",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        points=False,
    )
    style_fig(fig, height=180, title="Age by segment")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
with r3[3]:
    fig = px.box(
        df_plot,
        x="Segment",
        y="Days_is_client",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        points=False,
    )
    style_fig(fig, height=180, title="Tenure (days)")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

# ROW 5: 3 insight pills
from utils import pill

p1, p2, p3 = st.columns(3)
with p1:
    pill(
        f"<b>👑 Champions</b> · {n_champ} cust ({n_champ / total_customers * 100:.1f}%) — disproportionate revenue. Top retention priority."
    )
with p2:
    pill(
        f"<b>💎 Loyal VIPs</b> · {n_loyal:,} cust — the revenue backbone. High AOV, consistent spend."
    )
with p3:
    pill(
        f"<b>💸 Price-Sensitive</b> · {n_ps:,} cust — automate, don't subsidize. Low unit economics."
    )
