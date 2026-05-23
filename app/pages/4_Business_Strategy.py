"""Page 5 — Business Strategy. Tight grid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    CLUSTERS,
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
    page_title="Strategy | Customer Segmentation",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

df = load_clustered_data()
total_rev = df["Expenses"].sum()

rev_by = (
    df.groupby("Cluster")
    .agg(Expenses=("Expenses", "sum"), Customers=("Cluster", "count"))
    .reset_index()
)
rev_by["Segment"] = rev_by["Cluster"].map(CLUSTER_NAMES)
rev_by["ARPU"] = rev_by["Expenses"] / rev_by["Customers"]

champ_share = rev_by.loc[rev_by["Cluster"] == 0, "Expenses"].iloc[0] / total_rev * 100
loyal_share = rev_by.loc[rev_by["Cluster"] == 2, "Expenses"].iloc[0] / total_rev * 100
ps_share = rev_by.loc[rev_by["Cluster"] == 1, "Expenses"].iloc[0] / total_rev * 100
hv_share = champ_share + loyal_share
hv_cust_pct = (
    (
        rev_by.loc[rev_by["Cluster"] == 0, "Customers"].iloc[0]
        + rev_by.loc[rev_by["Cluster"] == 2, "Customers"].iloc[0]
    )
    / len(df)
    * 100
)

hcol, k1, k2, k3, k4, k5 = st.columns([2.6, 1, 1, 1, 1, 1])
with hcol:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "💼 Business Strategy</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "Per-segment playbook · revenue framing</div>",
        unsafe_allow_html=True,
    )
with k1:
    mini_kpi(
        "Total rev",
        f"${total_rev / 1000:,.0f}K",
        "all segments",
        gradient="linear-gradient(135deg, #10B981 0%, #14B8A6 100%)",
    )
with k2:
    mini_kpi(
        "Champ rev",
        f"{champ_share:.1f}%",
        f"{(df['Cluster'] == 0).sum()} cust",
        gradient="linear-gradient(135deg, #F59E0B 0%, #F97316 100%)",
    )
with k3:
    mini_kpi(
        "VIP rev",
        f"{loyal_share:.1f}%",
        f"{(df['Cluster'] == 2).sum():,} cust",
        gradient="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
    )
with k4:
    mini_kpi(
        "P-S rev",
        f"{ps_share:.1f}%",
        f"{(df['Cluster'] == 1).sum():,} cust",
        gradient="linear-gradient(135deg, #EF4444 0%, #EC4899 100%)",
    )
with k5:
    mini_kpi("80/20", f"{hv_share:.0f}%", f"from {hv_cust_pct:.0f}% cust")

# ROW 1: 4 charts
r1 = st.columns(4)

with r1[0]:
    fig = px.pie(
        rev_by,
        values="Expenses",
        names="Segment",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        hole=0.6,
    )
    fig.update_traces(textinfo="percent", textfont=dict(size=10, color="white"))
    style_fig(fig, height=220, show_legend=True, title="Revenue share")
    fig.update_layout(
        annotations=[
            dict(
                text=f"${total_rev / 1000:.0f}K",
                x=0.5,
                y=0.5,
                font=dict(size=14),
                showarrow=False,
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

with r1[1]:
    fig = px.pie(
        rev_by,
        values="Customers",
        names="Segment",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        hole=0.6,
    )
    fig.update_traces(textinfo="percent", textfont=dict(size=10, color="white"))
    style_fig(fig, height=220, show_legend=True, title="Customer share")
    fig.update_layout(
        annotations=[
            dict(text=f"{len(df):,}", x=0.5, y=0.5, font=dict(size=14), showarrow=False)
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

with r1[2]:
    fig = px.bar(
        rev_by,
        x="Segment",
        y="ARPU",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        text=rev_by["ARPU"].apply(lambda v: f"${v:,.0f}"),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10))
    style_fig(fig, height=220, title="ARPU (avg per customer)")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

with r1[3]:
    cmp_data = df.groupby("Cluster")["TotalAcceptedCmp"].mean().reset_index()
    cmp_data["Segment"] = cmp_data["Cluster"].map(CLUSTER_NAMES)
    fig = px.bar(
        cmp_data,
        x="Segment",
        y="TotalAcceptedCmp",
        color="Segment",
        color_discrete_map=COLOR_MAP_BY_NAME,
        text=cmp_data["TotalAcceptedCmp"].apply(lambda v: f"{v:.2f}"),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10))
    style_fig(fig, height=220, title="Avg campaigns accepted")
    fig.update_xaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

# ROW 2: 3 playbook cards
PLAYBOOK = {
    0: {
        "prio": "PROTECT",
        "budget": "Premium · 1:1",
        "channel": "Account managers",
        "actions": [
            "White-glove mgmt",
            "Early access",
            "Custom offers",
            "Referrals",
            "Feedback loop",
        ],
        "kpi": "NPS · retention · referrals",
        "warn": "Lose 5 = noticeable dent",
    },
    1: {
        "prio": "RETAIN CHEAPLY",
        "budget": "Low · automation",
        "channel": "Email · SMS · retargeting",
        "actions": [
            "Starter packs",
            "Time-limited disc.",
            "Drip onboarding",
            "Brand familiarity",
            "Win-back triggers",
        ],
        "kpi": "Cost/eng · graduation rate",
        "warn": "Premium channels don't pay back",
    },
    2: {
        "prio": "GROW SPEND",
        "budget": "Med-high · best ROI",
        "channel": "Segmented email + web",
        "actions": [
            "Loyalty tiers",
            "Premium bundles",
            "Personalized recs",
            "VIP previews",
            "Pathway to Champion",
        ],
        "kpi": "AOV · frequency · graduation",
        "warn": "Disengagement here = costliest mistake",
    },
}

cols = st.columns(3)
order = [0, 2, 1]
for col, cid in zip(cols, order):
    info = CLUSTERS[cid]
    plan = PLAYBOOK[cid]
    n = (df["Cluster"] == cid).sum()
    rs = rev_by.loc[rev_by["Cluster"] == cid, "Expenses"].iloc[0] / total_rev * 100
    actions_html = "".join(
        f'<li style="margin-bottom:2px;">{a}</li>' for a in plan["actions"]
    )
    with col:
        st.markdown(
            f'<div style="border:2px solid {info["color"]}; border-radius:10px; overflow:hidden;">'
            f'<div style="background:{info["gradient"]}; color:white; padding:10px 12px;">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<div style="font-size:22px;">{info["icon"]}</div>'
            f"<div>"
            f'<div style="font-size:14px; font-weight:800;">{info["name"]}</div>'
            f'<div style="font-size:10px; opacity:0.9;">{n:,} · {rs:.1f}% rev</div>'
            f"</div></div>"
            f'<div style="margin-top:6px; background:rgba(255,255,255,0.2); padding:3px 8px; '
            f'border-radius:5px; font-size:9px; font-weight:700; letter-spacing:0.06em; display:inline-block;">'
            f"⚡ {plan['prio']}</div></div>"
            f'<div style="padding:10px 12px; background:white;">'
            f'<div style="font-size:9px; color:#6B7280; font-weight:600;">💰 BUDGET</div>'
            f'<div style="font-size:11px; color:#1A1F2E; margin-bottom:4px;">{plan["budget"]}</div>'
            f'<div style="font-size:9px; color:#6B7280; font-weight:600;">📡 CHANNEL</div>'
            f'<div style="font-size:11px; color:#1A1F2E; margin-bottom:4px;">{plan["channel"]}</div>'
            f'<div style="font-size:9px; color:#6B7280; font-weight:600;">🎯 ACTIONS</div>'
            f'<ul style="font-size:10px; color:#1A1F2E; padding-left:16px; margin:2px 0 4px 0;">{actions_html}</ul>'
            f'<div style="font-size:9px; color:#6B7280; font-weight:600;">📊 KPI</div>'
            f'<div style="font-size:10px; color:#1A1F2E; margin-bottom:6px;">{plan["kpi"]}</div>'
            f'<div style="background:{info["color_light"]}; border-radius:5px; padding:5px 8px; '
            f'font-size:10px; color:{info["color"]}; font-weight:600;">⚠️ {plan["warn"]}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

# ROW 3: 4 next-step cards
nxt = [
    ("📉", "Churn prediction", "Per-segment supervised"),
    ("🧪", "A/B testing", "Measure lift per strategy"),
    ("🔄", "Re-segmentation", "Refit monthly"),
    ("💰", "CLV scoring", "Segment + lifetime value"),
]
r3 = st.columns(4)
for col, (icon, title, desc) in zip(r3, nxt):
    with col:
        st.markdown(
            f'<div style="background:#F7F9FC; border:1px solid #E5E7EB; border-radius:8px; padding:8px 12px;">'
            f'<div style="font-size:16px;">{icon}</div>'
            f'<div style="font-size:11px; font-weight:700; color:#1A1F2E;">{title}</div>'
            f'<div style="font-size:9px; color:#6B7280;">{desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
