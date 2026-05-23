"""Page 4 — Predict My Segment. Tight grid."""

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
    load_models,
    load_tsne,
    predict_cluster,
    project_to_tsne,
    render_sidebar,
    style_fig,
    pill,
)

st.set_page_config(
    page_title="Predict | Customer Segmentation",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

scaler, pca, kmeans, feature_cols = load_models()
df = load_clustered_data()
tsne_df = load_tsne()

st.sidebar.markdown("### 🎛️ Customer profile")
with st.sidebar.form("predict_form"):
    age = st.slider("Age", 25, 80, 45)
    income = st.slider("Income $", 5000, 120000, 50000, step=1000)
    days_client = st.slider("Days client", 0, 800, 400)
    recency = st.slider("Recency", 0, 100, 30)
    kids = st.selectbox("Kids", [0, 1, 2, 3], index=1)
    expenses = st.slider("Expenses $", 0, 2500, 600, step=50)
    purchases = st.slider("Purchases", 0, 50, 12)
    accepted_cmp = st.slider("Cmp accepted", 0, 4, 0)
    response = st.selectbox(
        "Last response", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    education = st.selectbox("Education", ["Graduate", "Undergraduate", "Postgraduate"])
    marital = st.selectbox("Marital", ["Partner", "Single"])
    submitted = st.form_submit_button(
        "🔮 Predict", use_container_width=True, type="primary"
    )


def build_input() -> dict:
    return {
        "Age": age,
        "Income": income,
        "Days_is_client": days_client,
        "Recency": recency,
        "Expenses": expenses,
        "TotalNumPurchases": purchases,
        "Complain": 0,
        "Response": response,
        "Education_Undergraduate": 1 if education == "Undergraduate" else 0,
        "Education_Postgraduate": 1 if education == "Postgraduate" else 0,
        "Kids_1": 1 if kids == 1 else 0,
        "Kids_2": 1 if kids == 2 else 0,
        "Kids_3": 1 if kids == 3 else 0,
        "TotalAcceptedCmp_1": 1 if accepted_cmp == 1 else 0,
        "TotalAcceptedCmp_2": 1 if accepted_cmp == 2 else 0,
        "TotalAcceptedCmp_3": 1 if accepted_cmp == 3 else 0,
        "TotalAcceptedCmp_4": 1 if accepted_cmp >= 4 else 0,
        "Marital_Status_Single": 1 if marital == "Single" else 0,
    }


if not submitted and "last_prediction" not in st.session_state:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "🔮 Predict My Segment</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "Fill the form on the left → live K-Means inference + tailored strategy</div>",
        unsafe_allow_html=True,
    )
    st.info("👈 Set the profile on the left and click Predict.")
    st.stop()

if submitted:
    user_input = build_input()
    cluster_id, pca_coords = predict_cluster(user_input)
    tsne_xy = project_to_tsne(pca_coords)
    st.session_state["last_prediction"] = {
        "cluster_id": cluster_id,
        "pca_coords": pca_coords,
        "tsne_xy": tsne_xy,
        "user_input": user_input,
    }

pred = st.session_state["last_prediction"]
cluster_id = pred["cluster_id"]
info = CLUSTERS[cluster_id]
seg_df = df[df["Cluster"] == cluster_id]
ui = pred["user_input"]

hcol, banner_col = st.columns([1.5, 2.5])
with hcol:
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #6366F1, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        "🔮 Predict My Segment</h1>"
        '<div style="font-size: 11px; color: #6B7280; margin-top: -2px;">'
        "Live K-Means inference</div>",
        unsafe_allow_html=True,
    )
with banner_col:
    st.markdown(
        f'<div style="background: {info["gradient"]}; color: white; '
        f'padding: 12px 18px; border-radius: 10px;">'
        f'<div style="display:flex; align-items:center; gap:14px;">'
        f'<div style="font-size:32px;">{info["icon"]}</div>'
        f"<div>"
        f'<div style="font-size:10px; opacity:0.9; text-transform:uppercase; letter-spacing:0.1em;">Predicted</div>'
        f'<div style="font-size:18px; font-weight:800;">{info["long_name"]}</div>'
        f'<div style="font-size:11px; opacity:0.95;">{info["tagline"]}</div>'
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    mini_kpi(
        "Your income",
        f"${ui['Income']:,}",
        f"Seg med: ${seg_df['Income'].median():,.0f}",
        gradient=info["gradient"],
    )
with k2:
    mini_kpi(
        "Your spend",
        f"${ui['Expenses']:,}",
        f"Seg med: ${seg_df['Expenses'].median():,.0f}",
    )
with k3:
    mini_kpi(
        "Your purch.",
        f"{ui['TotalNumPurchases']}",
        f"Seg med: {seg_df['TotalNumPurchases'].median():.0f}",
    )
with k4:
    mini_kpi("Your age", f"{ui['Age']}", f"Seg med: {seg_df['Age'].median():.0f}")
with k5:
    mini_kpi(
        "Segment size",
        f"{len(seg_df):,}",
        f"{len(seg_df) / len(df) * 100:.1f}% of base",
    )

r1c1, r1c2 = st.columns([1.2, 1])

with r1c1:
    fig = px.scatter(
        tsne_df,
        x="tsne_x",
        y="tsne_y",
        color=tsne_df["Cluster"].map(CLUSTER_NAMES),
        color_discrete_map=COLOR_MAP_BY_NAME,
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.add_trace(
        go.Scatter(
            x=[pred["tsne_xy"][0]],
            y=[pred["tsne_xy"][1]],
            mode="markers",
            name="You",
            marker=dict(
                size=24,
                color="#FFFFFF",
                line=dict(width=4, color=info["color"]),
                symbol="star",
            ),
        )
    )
    style_fig(fig, height=280, show_legend=True, title="t-SNE map — ⭐ = your customer")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    metrics = ["Income", "Expenses", "TotalNumPurchases", "Age"]
    rows = []
    for m in metrics:
        rows.append({"Metric": m, "Group": "You", "Value": ui[m]})
        for cid in CLUSTERS:
            sub = df[df["Cluster"] == cid]
            rows.append(
                {
                    "Metric": m,
                    "Group": CLUSTERS[cid]["name"],
                    "Value": float(sub[m].median()),
                }
            )
    comp_df = pd.DataFrame(rows)
    color_map = {
        "You": "#1F2937",
        **{CLUSTERS[k]["name"]: CLUSTERS[k]["color"] for k in CLUSTERS},
    }
    fig = px.bar(
        comp_df,
        x="Metric",
        y="Value",
        color="Group",
        barmode="group",
        color_discrete_map=color_map,
    )
    style_fig(fig, height=280, show_legend=True, title="You vs segment medians")
    st.plotly_chart(fig, use_container_width=True)

STRATEGIES = {
    0: [
        ("👑", "White-glove", "Dedicated account manager."),
        ("🎁", "Early access", "First dibs on launches."),
        ("📊", "Custom offers", "Built around their behavior."),
        ("🤝", "Referrals", "Brand champions bring friends."),
    ],
    1: [
        ("🎁", "Starter packs", "Lower price barrier."),
        ("📱", "Automation only", "Email/SMS, no premium."),
        ("⏰", "Time-limited", "Urgency-based pricing."),
        ("🌱", "Onboarding", "Build familiarity cheaply."),
    ],
    2: [
        ("🏆", "Loyalty tiers", "Reward consistent spend."),
        ("📈", "Premium upsell", "High-margin SKUs."),
        ("💌", "Personalized recs", "Drive AOV growth."),
        ("🎟️", "VIP events", "Strengthen relationship."),
    ],
}
cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, STRATEGIES[cluster_id]):
    with col:
        st.markdown(
            f'<div style="background:{info["color_light"]}; border-left:3px solid {info["color"]}; '
            f'border-radius:8px; padding:10px 12px; height:80px;">'
            f'<div style="font-size:18px;">{icon}</div>'
            f'<div style="font-size:12px; font-weight:700; color:{info["color"]}; margin-top:2px;">{title}</div>'
            f'<div style="font-size:10px; color:#4B5563; margin-top:2px; line-height:1.3;">{desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
