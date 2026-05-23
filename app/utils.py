"""Shared utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

PALETTE = {
    "bg": "#FFFFFF",
    "panel": "#F7F9FC",
    "text": "#1A1F2E",
    "muted": "#6B7280",
    "border": "#E5E7EB",
}

CLUSTERS = {
    0: {
        "name": "Champions",
        "long_name": "Champions — Campaign Responders",
        "color": "#F59E0B",
        "color_light": "#FEF3C7",
        "gradient": "linear-gradient(135deg, #F59E0B 0%, #F97316 100%)",
        "tagline": "Tiny elite: highest income, biggest spenders.",
        "icon": "👑",
    },
    1: {
        "name": "Price-Sensitive",
        "long_name": "Price-Sensitive — Low-Value",
        "color": "#EF4444",
        "color_light": "#FEE2E2",
        "gradient": "linear-gradient(135deg, #EF4444 0%, #EC4899 100%)",
        "tagline": "Low income, infrequent purchases.",
        "icon": "💸",
    },
    2: {
        "name": "Loyal VIPs",
        "long_name": "Loyal VIPs — High-Value Regulars",
        "color": "#6366F1",
        "color_light": "#E0E7FF",
        "gradient": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
        "tagline": "High income, steady spend.",
        "icon": "💎",
    },
}
CLUSTER_COLORS = {k: v["color"] for k, v in CLUSTERS.items()}
CLUSTER_NAMES = {k: v["name"] for k, v in CLUSTERS.items()}
COLOR_MAP_BY_NAME = {v["name"]: v["color"] for v in CLUSTERS.values()}


GLOBAL_CSS = """
<style>
    .main .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }
    h1 { font-size: 1.45rem !important; font-weight: 800 !important; margin: 0 0 0.1rem 0 !important; }
    h2 { font-size: 1.05rem !important; font-weight: 700 !important; margin: 0.3rem 0 !important; }
    h3 { font-size: 0.85rem !important; font-weight: 700 !important; }
    p, div { font-size: 13px; }
    .mini-kpi {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 8px 12px;
        text-align: left;
    }
    .mini-kpi.grad {
        background: var(--grad);
        color: white;
        border: none;
    }
    .mini-kpi .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.75; font-weight: 600; line-height: 1.1; }
    .mini-kpi .val { font-size: 22px; font-weight: 800; line-height: 1.1; margin-top: 2px; }
    .mini-kpi .sub { font-size: 10px; opacity: 0.75; margin-top: 1px; line-height: 1.2; }
    .seg-chip {
        border-radius: 10px;
        padding: 8px 12px;
        color: white;
        font-size: 11px;
    }
    .seg-chip .nm { font-size: 14px; font-weight: 800; }
    .seg-chip .st { font-size: 10px; opacity: 0.9; margin-top: 1px; }
    .pill {
        font-size: 11px;
        background: #F3F4F6;
        color: #374151;
        padding: 6px 10px;
        border-radius: 6px;
        border-left: 3px solid #6366F1;
        line-height: 1.4;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    .stDeployButton { display: none; }
    .element-container { margin-bottom: 0.3rem !important; }
    [data-testid="stVerticalBlock"] > div { gap: 0.4rem !important; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def style_fig(
    fig: go.Figure, height: int = 180, show_legend: bool = False, title: str = None
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24 if title else 6, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, sans-serif", size=10, color=PALETTE["text"]
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        xaxis=dict(
            gridcolor="#F1F3F5",
            zerolinecolor="#F1F3F5",
            title=None,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            gridcolor="#F1F3F5",
            zerolinecolor="#F1F3F5",
            title=None,
            tickfont=dict(size=9),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=11), y=0.96)
        )
    return fig


@st.cache_resource
def load_models():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(MODELS_DIR / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return scaler, pca, kmeans, feature_cols


@st.cache_data
def load_clustered_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "clustered_data.parquet")


@st.cache_data
def load_profiles() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cluster_profiles.parquet")


@st.cache_data
def load_tsne() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "tsne_coords.parquet")


def predict_cluster(user_input: dict) -> tuple[int, np.ndarray]:
    scaler, pca, kmeans, feature_cols = load_models()
    row = {col: user_input.get(col, 0) for col in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return int(kmeans.predict(X_pca)[0]), X_pca[0]


def project_to_tsne(pca_coords: np.ndarray) -> tuple[float, float]:
    tsne_df = load_tsne()
    scaler, pca, kmeans, _ = load_models()
    cid = int(kmeans.predict(pca_coords.reshape(1, -1))[0])
    sub = tsne_df[tsne_df["Cluster"] == cid]
    return float(sub["tsne_x"].median()), float(sub["tsne_y"].median())


def mini_kpi(label: str, value: str, sub: str = "", gradient: str = None):
    style_var = f'style="--grad: {gradient};"' if gradient else ""
    cls = "mini-kpi grad" if gradient else "mini-kpi"
    st.markdown(
        f'<div class="{cls}" {style_var}>'
        f'<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def seg_chip(cluster_id: int, n: int, total: int):
    c = CLUSTERS[cluster_id]
    pct = n / total * 100
    st.markdown(
        f'<div class="seg-chip" style="background: {c["gradient"]};">'
        f'<div class="nm">{c["icon"]} {c["name"]}</div>'
        f'<div class="st">{n:,} · {pct:.1f}%</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def pill(text: str):
    st.markdown(f'<div class="pill">{text}</div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); '
            'padding: 14px; border-radius: 10px; color: white; margin-bottom: 12px;">'
            '<div style="font-size: 14px; font-weight: 800;">Gheffari Nour El Houda</div>'
            '<div style="font-size: 11px; opacity: 0.9;">Data Scientist · NLP · Blida, Algeria</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "🔗 [GitHub](https://github.com/houdhoudGH) · "
            "📧 [Email](mailto:nourgheffari@gmail.com)"
        )
        st.divider()
        st.caption("**Customer Segmentation**")
        st.caption("K-Means · PCA · t-SNE · 2,198 customers")
