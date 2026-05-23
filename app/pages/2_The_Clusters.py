"""Page 3 — The Clusters. Tight grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    pill,
)

st.set_page_config(
    page_title="Clusters | Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

df = load_clustered_data()
profiles = load_profiles()
tsne_df = load_tsne()
prof = profiles.reset_index().copy()
prof["Segment"] = prof["Cluster"].map(CLUSTER_NAMES)

hcol, k1, k2, k3, k4, k5 = st.columns([2.6, 1, 1, 1, 1, 1])
with hcol:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "🎯 The Three Segments</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "K-Means · k=3 · PCA 10 components · 80% variance</div>",
        unsafe_allow_html=True,
    )
with k1:
    mini_kpi(
        "PCA dim",
        "10",
        "80% var",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k2:
    mini_kpi("Optimal k", "3", "elbow")
with k3:
    mini_kpi(
        "Champions",
        f"{(df['Cluster'] == 0).sum()}",
        "1.9%",
        gradient="linear-gradient(135deg, #F59E0B 0%, #F97316 100%)",
    )
with k4:
    mini_kpi(
        "Loyal VIPs",
        f"{(df['Cluster'] == 2).sum():,}",
        "44.4%",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k5:
    mini_kpi(
        "Price-Sens.",
        f"{(df['Cluster'] == 1).sum():,}",
        "53.7%",
        gradient="linear-gradient(135deg, #EF4444 0%, #EC4899 100%)",
    )

# ROW 1: t-SNE | Radar | chips
r1c1, r1c2, r1c3 = st.columns([1.3, 1.1, 0.7])

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
    style_fig(fig, height=300, show_legend=True, title="t-SNE projection")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    radar_m = ["Income", "Expenses", "TotalNumPurchases", "Age", "Days_is_client"]
    norm = profiles[radar_m].copy()
    for c in radar_m:
        norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
    fig = go.Figure()
    for cid in profiles.index:
        c = CLUSTERS[cid]
        fig.add_trace(
            go.Scatterpolar(
                r=norm.loc[cid].tolist() + [norm.loc[cid].iloc[0]],
                theta=radar_m + [radar_m[0]],
                fill="toself",
                name=c["name"],
                line=dict(color=c["color"], width=2),
                fillcolor=c["color"],
                opacity=0.3,
            )
        )
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#F7F9FC",
            radialaxis=dict(
                visible=True, range=[0, 1], showticklabels=False, gridcolor="#E5E7EB"
            ),
            angularaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=8)),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=9)
        ),
        margin=dict(l=30, r=30, t=24, b=20),
        title=dict(text="<b>Segment fingerprints</b>", font=dict(size=11), y=0.96),
    )
    st.plotly_chart(fig, use_container_width=True)

with r1c3:
    for cid in [0, 2, 1]:
        seg_chip(cid, int((df["Cluster"] == cid).sum()), len(df))
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

# ROW 2: 4 bars
r2 = st.columns(4)
charts = [
    ("Income", "Income median", "$"),
    ("Expenses", "Spend median", "$"),
    ("TotalNumPurchases", "Purchases", ""),
    ("TotalAcceptedCmp", "Campaigns acc.", ""),
]
for col, (m, title, pre) in zip(r2, charts):
    with col:
        fig = px.bar(
            prof,
            x="Segment",
            y=m,
            color="Segment",
            color_discrete_map=COLOR_MAP_BY_NAME,
            text=prof[m].apply(
                lambda v: f"{pre}{v:,.1f}" if pre == "" else f"{pre}{v:,.0f}"
            ),
        )
        fig.update_traces(textposition="outside", textfont=dict(size=9))
        style_fig(fig, height=180, title=title)
        fig.update_xaxes(tickfont=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

# ROW 3: 4 boxes
df_plot = df.copy()
df_plot["Segment"] = df_plot["Cluster"].map(CLUSTER_NAMES)

r3 = st.columns(4)
box_charts = [
    ("Income", "Income spread"),
    ("Expenses", "Spend spread"),
    ("Age", "Age spread"),
    ("Days_is_client", "Tenure spread"),
]
for col, (m, title) in zip(r3, box_charts):
    with col:
        fig = px.box(
            df_plot,
            x="Segment",
            y=m,
            color="Segment",
            color_discrete_map=COLOR_MAP_BY_NAME,
            points=False,
        )
        style_fig(fig, height=180, title=title)
        fig.update_xaxes(tickfont=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

# ROW 4: pipeline pills
steps = [
    ("1·Clean", "#6366F1"),
    ("2·Engineer", "#7C3AED"),
    ("3·Encode", "#A855F7"),
    ("4·Scale", "#EC4899"),
    ("5·PCA", "#F59E0B"),
    ("6·K-Means", "#10B981"),
    ("7·t-SNE", "#14B8A6"),
]
cols = st.columns(len(steps))
for col, (title, color) in zip(cols, steps):
    with col:
        st.markdown(
            f'<div style="background:{color}; color:white; border-radius:8px; padding:8px 6px; text-align:center; font-size:11px; font-weight:700;">{title}</div>',
            unsafe_allow_html=True,
        )
