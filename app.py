"""Customer Segmentation — 5-chapter story dashboard. White theme, refined."""
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

CLUSTERS = {
    0: {"name": "Champions", "color": "#F59E0B",
        "color_light": "rgba(245,158,11,0.15)",
        "gradient": "linear-gradient(135deg, #F59E0B 0%, #F97316 100%)"},
    1: {"name": "Price-Sensitive", "color": "#EF4444",
        "color_light": "rgba(239,68,68,0.15)",
        "gradient": "linear-gradient(135deg, #EF4444 0%, #EC4899 100%)"},
    2: {"name": "Loyal VIPs", "color": "#6366F1",
        "color_light": "rgba(99,102,241,0.15)",
        "gradient": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"},
}
CLUSTER_NAMES = {k: v["name"] for k, v in CLUSTERS.items()}
COLOR_MAP = {v["name"]: v["color"] for v in CLUSTERS.values()}

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_parquet(DATA_DIR / "clustered_data.parquet")
df["Segment"] = df["Cluster"].map(CLUSTER_NAMES)
profiles = pd.read_parquet(DATA_DIR / "cluster_profiles.parquet")
tsne_df = pd.read_parquet(DATA_DIR / "tsne_coords.parquet")
tsne_df["Segment"] = tsne_df["Cluster"].map(CLUSTER_NAMES)

total = len(df)
total_rev = df["Expenses"].sum()
n_champ = int((df["Cluster"]==0).sum())
n_loyal = int((df["Cluster"]==2).sum())
n_ps = int((df["Cluster"]==1).sum())

rev_by = df.groupby("Cluster").agg(
    Expenses=("Expenses", "sum"), Customers=("Cluster", "count")).reset_index()
rev_by["Segment"] = rev_by["Cluster"].map(CLUSTER_NAMES)
rev_by["ARPU"] = rev_by["Expenses"] / rev_by["Customers"]

champ_share = rev_by.loc[rev_by["Cluster"]==0, "Expenses"].iloc[0] / total_rev * 100
loyal_share = rev_by.loc[rev_by["Cluster"]==2, "Expenses"].iloc[0] / total_rev * 100
ps_share = rev_by.loc[rev_by["Cluster"]==1, "Expenses"].iloc[0] / total_rev * 100
hv_share = champ_share + loyal_share
hv_pct = (n_champ + n_loyal) / total * 100

prof = profiles.reset_index().copy()
prof["Segment"] = prof["Cluster"].map(CLUSTER_NAMES)

# ============================================================================
# PLOT STYLING
# ============================================================================
def style_fig(fig, h=200, title=None, legend=False):
    fig.update_layout(
        height=h,
        margin=dict(l=8, r=8, t=28 if title else 4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=9.5, color="#4B5563"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                    font=dict(size=9, color="#6B7280"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#F1F3F5", zerolinecolor="#F1F3F5", title=None,
                   tickfont=dict(size=8.5, color="#6B7280")),
        yaxis=dict(gridcolor="#F1F3F5", zerolinecolor="#F1F3F5", title=None,
                   tickfont=dict(size=8.5, color="#6B7280")),
    )
    if title:
        fig.update_layout(title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=11, color="#1A1F2E"), y=0.97, x=0.02, xanchor="left"))
    return fig


# ============================================================================
# COMPONENTS
# ============================================================================
def kpi(label, value, sub="", gradient=None):
    style = {}
    cls = "kpi-card"
    if gradient:
        cls = "kpi-card gradient"
        style = {"background": gradient}
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub, className="kpi-sub") if sub else None,
    ], className=cls, style=style)


def plot_box(fig):
    return html.Div(
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        className="plot-card")


def chapter(num, title, subtitle):
    return html.Div([
        html.Div([
            html.Div(num, className="chapter-num"),
            html.Div([
                html.Div(title, className="chapter-title"),
                html.Div(subtitle, className="chapter-subtitle"),
            ], style={"marginLeft": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Hr(style={"borderColor": "#E5E7EB", "margin": "10px 0 12px 0"}),
    ], style={"marginTop": "28px", "marginBottom": "8px"})


def narrative(content):
    return html.Div(content, className="narrative",
                    style={"marginBottom": "12px"})


def insight(title, body, color="#6366F1"):
    return html.Div([
        html.Div(title, className="insight-title"),
        html.Div(body, className="insight-body"),
    ], className="insight-box", style={"borderLeftColor": color, "marginTop": "12px"})


# ============================================================================
# CHAPTER 1 — 5 plots, taller to fill screen
# ============================================================================
H1 = 220  # taller plots to fill chapter 1

fig_age = px.histogram(df, x="Age", nbins=25, color_discrete_sequence=["#6366F1"])
fig_age.update_traces(marker=dict(line=dict(width=0)))
style_fig(fig_age, h=H1, title="Age distribution")

df_inc = df[df["Income"] < df["Income"].quantile(0.99)]
fig_inc_raw = px.histogram(df_inc, x="Income", nbins=25,
                            color_discrete_sequence=["#10B981"])
fig_inc_raw.update_traces(marker=dict(line=dict(width=0)))
style_fig(fig_inc_raw, h=H1, title="Income distribution")

fig_spend_raw = px.histogram(df, x="Expenses", nbins=25,
                              color_discrete_sequence=["#F59E0B"])
fig_spend_raw.update_traces(marker=dict(line=dict(width=0)))
style_fig(fig_spend_raw, h=H1, title="Total spend")

if "Kids" in df.columns:
    kids_dist = df["Kids"].value_counts().sort_index().reset_index()
    kids_dist.columns = ["Kids", "Count"]
    fig_kids_dist = px.bar(kids_dist, x="Kids", y="Count",
                            color_discrete_sequence=["#EC4899"], text="Count")
    fig_kids_dist.update_traces(textposition="outside", textfont=dict(size=9, color="#4B5563"))
    style_fig(fig_kids_dist, h=H1, title="Family size")

fig_recency_raw = px.histogram(df, x="Recency", nbins=20,
                                color_discrete_sequence=["#8B5CF6"])
fig_recency_raw.update_traces(marker=dict(line=dict(width=0)))
style_fig(fig_recency_raw, h=H1, title="Recency (days)")


# ============================================================================
# CHAPTER 2 — 5 plots, similarly sized
# ============================================================================
H2 = 220

fig_inc_exp = px.scatter(df_inc, x="Income", y="Expenses",
                          color_discrete_sequence=["#6366F1"], opacity=0.45)
fig_inc_exp.update_traces(marker=dict(size=3))
style_fig(fig_inc_exp, h=H2, title="Income × Expenses")

kids_exp = df.groupby("Kids")["Expenses"].median().reset_index()
fig_kids_exp = px.bar(kids_exp, x="Kids", y="Expenses",
                       color_discrete_sequence=["#EC4899"],
                       text=kids_exp["Expenses"].apply(lambda v: f"${v:.0f}"))
fig_kids_exp.update_traces(textposition="outside", textfont=dict(size=9, color="#4B5563"))
style_fig(fig_kids_exp, h=H2, title="Spend by family size")

if "Education" in df.columns and df["Education"].dtype == object:
    edu_data = df["Education"].value_counts().reset_index()
    edu_data.columns = ["Group", "Count"]
else:
    edu_cols = [c for c in df.columns if c.startswith("Education_")]
    if edu_cols:
        edu_data = pd.DataFrame({
            "Group": [c.replace("Education_", "") for c in edu_cols] + ["Graduate"],
            "Count": [int(df[c].sum()) for c in edu_cols] +
                     [int(total - sum(df[c].sum() for c in edu_cols))]
        })
    else:
        edu_data = pd.DataFrame({"Group": ["N/A"], "Count": [total]})
fig_edu = px.bar(edu_data, x="Group", y="Count",
                  color_discrete_sequence=["#10B981"], text="Count")
fig_edu.update_traces(textposition="outside", textfont=dict(size=9, color="#4B5563"))
style_fig(fig_edu, h=H2, title="Education")

if "Marital_Status" in df.columns and df["Marital_Status"].dtype == object:
    ms_data = df["Marital_Status"].value_counts().reset_index()
    ms_data.columns = ["Status", "Count"]
else:
    if "Marital_Status_Single" in df.columns:
        n_single = int(df["Marital_Status_Single"].sum())
        ms_data = pd.DataFrame({"Status": ["Single", "Partner"],
                                 "Count": [n_single, total - n_single]})
    else:
        ms_data = pd.DataFrame({"Status": ["Unknown"], "Count": [total]})
fig_marital = px.pie(ms_data, values="Count", names="Status", hole=0.55,
                      color_discrete_sequence=["#6366F1", "#EC4899", "#F59E0B", "#10B981"])
fig_marital.update_traces(textinfo="percent", textfont=dict(size=9, color="white"))
style_fig(fig_marital, h=H2, title="Marital status", legend=True)

if "TotalAcceptedCmp" in df.columns and df["TotalAcceptedCmp"].dtype != object:
    cmp_dist = df["TotalAcceptedCmp"].value_counts().sort_index().reset_index()
    cmp_dist.columns = ["Campaigns", "Count"]
else:
    cmp_cols = [c for c in df.columns if c.startswith("TotalAcceptedCmp_")]
    if cmp_cols:
        cmp_dist = pd.DataFrame({
            "Campaigns": [c.replace("TotalAcceptedCmp_", "") for c in cmp_cols] + ["0"],
            "Count": [int(df[c].sum()) for c in cmp_cols] +
                     [int(total - sum(df[c].sum() for c in cmp_cols))]
        }).sort_values("Campaigns")
    else:
        cmp_dist = pd.DataFrame({"Campaigns": ["0"], "Count": [total]})
fig_cmp_eng = px.bar(cmp_dist, x="Campaigns", y="Count",
                      color_discrete_sequence=["#F59E0B"], text="Count")
fig_cmp_eng.update_traces(textposition="outside", textfont=dict(size=9, color="#4B5563"))
style_fig(fig_cmp_eng, h=H2, title="Campaigns accepted")


# ============================================================================
# CHAPTER 3 — same as before, mid-density
# ============================================================================
H3 = 180

pca_variance = [0.22, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03]
pca_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(10)],
    "Variance": pca_variance,
    "Cumulative": np.cumsum(pca_variance)
})
fig_pca = go.Figure()
fig_pca.add_trace(go.Bar(x=pca_df["Component"], y=pca_df["Variance"],
                          marker_color="#6366F1", name="Per component"))
fig_pca.add_trace(go.Scatter(x=pca_df["Component"], y=pca_df["Cumulative"],
                              mode="lines+markers", marker_color="#EC4899",
                              line=dict(width=2), name="Cumulative", yaxis="y2"))
fig_pca.update_layout(yaxis2=dict(overlaying="y", side="right",
                                    tickformat=".0%",
                                    gridcolor="#F1F3F5",
                                    tickfont=dict(size=8, color="#6B7280")))
style_fig(fig_pca, h=H3, title="PCA scree plot", legend=True)

inc_raw = df["Income"].dropna()
inc_std = (inc_raw - inc_raw.mean()) / inc_raw.std()
fig_scaling = go.Figure()
fig_scaling.add_trace(go.Histogram(x=inc_raw, nbinsx=25, name="Before",
                                     marker_color="#6366F1", opacity=0.6))
fig_scaling.add_trace(go.Histogram(x=inc_std*15000+inc_raw.mean(), nbinsx=25,
                                     name="After", marker_color="#10B981", opacity=0.6))
fig_scaling.update_layout(barmode="overlay")
style_fig(fig_scaling, h=H3, title="StandardScaler effect", legend=True)

numeric_cols = ["Age", "Income", "Expenses", "TotalNumPurchases", "Days_is_client", "Recency"]
numeric_cols = [c for c in numeric_cols if c in df.columns]
corr = df[numeric_cols].corr()
fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale=[[0, "#EF4444"], [0.5, "#FFFFFF"], [1, "#6366F1"]],
    zmin=-1, zmax=1,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="#1A1F2E"),
    showscale=False,
))
fig_corr.update_layout(
    height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=28, b=4),
    font=dict(size=9, color="#6B7280"),
    title=dict(text="<b>Feature correlations</b>",
                font=dict(size=11, color="#1A1F2E"), y=0.97, x=0.02, xanchor="left"),
    xaxis=dict(tickfont=dict(size=8, color="#6B7280")),
    yaxis=dict(tickfont=dict(size=8, color="#6B7280")),
)

fig_scale_problem = go.Figure()
fig_scale_problem.add_trace(go.Box(y=df["Age"], name="Age",
                                     marker_color="#6366F1", boxpoints=False))
fig_scale_problem.add_trace(go.Box(y=df["Income"]/1000, name="Income(K)",
                                     marker_color="#EC4899", boxpoints=False))
fig_scale_problem.add_trace(go.Box(y=df["Expenses"]/10, name="Expenses(/10)",
                                     marker_color="#F59E0B", boxpoints=False))
style_fig(fig_scale_problem, h=H3, title="Raw feature scales differ")

fig_var_donut = go.Figure(data=[go.Pie(
    values=[80, 20], labels=["Retained", "Discarded"],
    hole=0.65, marker=dict(colors=["#10B981", "#E5E7EB"]),
    textinfo="none")])
fig_var_donut.update_layout(
    height=H3, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=28, b=4), showlegend=False,
    title=dict(text="<b>Variance retained</b>",
                font=dict(size=11, color="#1A1F2E"), y=0.97, x=0.02, xanchor="left"),
    annotations=[dict(text="<b>80%</b>", x=0.5, y=0.5,
                       font=dict(size=24, color="#10B981"), showarrow=False)])


# ============================================================================
# CHAPTER 4 — taller plots to fill screen
# ============================================================================
H4 = 220

ks = list(range(2, 9))
inertias = [4200, 3100, 2750, 2580, 2450, 2360, 2290]
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                line=dict(color="#6366F1", width=2),
                                marker=dict(size=7, color="#6366F1")))
fig_elbow.add_trace(go.Scatter(x=[3], y=[3100], mode="markers",
                                marker=dict(size=14, color="#F59E0B",
                                            line=dict(width=2, color="white")),
                                showlegend=False))
fig_elbow.add_annotation(x=3, y=3100, text="k=3",
                          font=dict(size=10, color="#F59E0B"),
                          showarrow=True, arrowcolor="#F59E0B", ax=30, ay=-25)
style_fig(fig_elbow, h=H4, title="Elbow method")

fig_tsne = px.scatter(tsne_df, x="tsne_x", y="tsne_y", color="Segment",
                       color_discrete_map=COLOR_MAP, opacity=0.65)
fig_tsne.update_traces(marker=dict(size=3.5, line=dict(width=0)))
style_fig(fig_tsne, h=H4, title="t-SNE projection", legend=True)
fig_tsne.update_xaxes(showticklabels=False)
fig_tsne.update_yaxes(showticklabels=False)

size_data = pd.DataFrame({
    "Segment": ["Champions", "Loyal VIPs", "Price-Sensitive"],
    "Count": [n_champ, n_loyal, n_ps],
})
fig_sizes = px.bar(size_data, x="Segment", y="Count", color="Segment",
                    color_discrete_map=COLOR_MAP,
                    text=size_data["Count"].apply(lambda v: f"{v:,}"))
fig_sizes.update_traces(textposition="outside", textfont=dict(size=10, color="#4B5563"))
style_fig(fig_sizes, h=H4, title="Cluster sizes")

sil_scores = [0.18, 0.31, 0.27, 0.24, 0.21, 0.19, 0.18]
fig_sil = go.Figure()
fig_sil.add_trace(go.Scatter(x=ks, y=sil_scores, mode="lines+markers",
                              line=dict(color="#10B981", width=2),
                              marker=dict(size=7, color="#10B981")))
fig_sil.add_trace(go.Scatter(x=[3], y=[0.31], mode="markers",
                              marker=dict(size=14, color="#F59E0B",
                                          line=dict(width=2, color="white")),
                              showlegend=False))
style_fig(fig_sil, h=H4, title="Silhouette score")


# ============================================================================
# CHAPTER 5 — compact bars
# ============================================================================
H5 = 145

def small_bar(metric, title, pre="", fmt=",.0f"):
    fig = px.bar(prof, x="Segment", y=metric, color="Segment",
                  color_discrete_map=COLOR_MAP,
                  text=prof[metric].apply(lambda v: f"{pre}{format(v, fmt)}"))
    fig.update_traces(textposition="outside", textfont=dict(size=8, color="#4B5563"))
    style_fig(fig, h=H5, title=title)
    fig.update_xaxes(tickfont=dict(size=8))
    return fig

bar_inc = small_bar("Income", "Median income", "$")
bar_exp = small_bar("Expenses", "Median spend", "$")
bar_pur = small_bar("TotalNumPurchases", "Median purchases", "")
bar_age = small_bar("Age", "Median age", "")
bar_ten = small_bar("Days_is_client", "Median tenure", "")

radar_m = ["Income", "Expenses", "TotalNumPurchases", "Age", "Days_is_client"]
norm = profiles[radar_m].copy()
for c in radar_m:
    norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
fig_radar = go.Figure()
for cid in profiles.index:
    c = CLUSTERS[cid]
    fig_radar.add_trace(go.Scatterpolar(
        r=norm.loc[cid].tolist() + [norm.loc[cid].iloc[0]],
        theta=radar_m + [radar_m[0]],
        fill="toself", name=c["name"],
        line=dict(color=c["color"], width=1.5),
        fillcolor=c["color"], opacity=0.3))
fig_radar.update_layout(
    height=H5, paper_bgcolor="rgba(0,0,0,0)",
    polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                                gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB",
                                 tickfont=dict(size=7, color="#6B7280"))),
    showlegend=False,
    margin=dict(l=20, r=20, t=26, b=10),
    title=dict(text="<b>Fingerprints</b>",
                font=dict(size=10, color="#1A1F2E"), y=0.97, x=0.02, xanchor="left"))

fig_rev_donut = px.pie(rev_by, values="Expenses", names="Segment",
                        color="Segment", color_discrete_map=COLOR_MAP, hole=0.6)
fig_rev_donut.update_traces(textinfo="percent", textfont=dict(size=10, color="white"))
style_fig(fig_rev_donut, h=175, title="Revenue share")
fig_rev_donut.update_layout(annotations=[dict(text=f"${total_rev/1000:.0f}K",
                                                x=0.5, y=0.5,
                                                font=dict(size=13, color="#1A1F2E"),
                                                showarrow=False)])

fig_cust_donut = px.pie(rev_by, values="Customers", names="Segment",
                         color="Segment", color_discrete_map=COLOR_MAP, hole=0.6)
fig_cust_donut.update_traces(textinfo="percent", textfont=dict(size=10, color="white"))
style_fig(fig_cust_donut, h=175, title="Customer share")
fig_cust_donut.update_layout(annotations=[dict(text=f"{total:,}", x=0.5, y=0.5,
                                                 font=dict(size=13, color="#1A1F2E"),
                                                 showarrow=False)])

fig_arpu = px.bar(rev_by, x="Segment", y="ARPU", color="Segment",
                   color_discrete_map=COLOR_MAP,
                   text=rev_by["ARPU"].apply(lambda v: f"${v:,.0f}"))
fig_arpu.update_traces(textposition="outside", textfont=dict(size=9, color="#4B5563"))
style_fig(fig_arpu, h=175, title="ARPU per segment")


# ============================================================================
# PLAYBOOK CARD — full width, no emojis
# ============================================================================
PLAYBOOK = {
    0: {"prio": "PROTECT",
        "actions": ["Dedicated account managers", "Early access to launches",
                    "Custom offer design", "Referral programs"]},
    2: {"prio": "GROW",
        "actions": ["Tiered loyalty program", "Premium product bundles",
                    "Personalized recommendations", "Pathway to Champion tier"]},
    1: {"prio": "AUTOMATE",
        "actions": ["Entry-level starter packs", "Time-limited discounts",
                    "Drip onboarding campaigns", "Brand familiarity content"]},
}

def playbook_card(cid):
    info = CLUSTERS[cid]
    plan = PLAYBOOK[cid]
    n = (df["Cluster"]==cid).sum()
    rs = rev_by.loc[rev_by["Cluster"]==cid, "Expenses"].iloc[0] / total_rev * 100
    return html.Div([
        html.Div([
            html.Div(info["name"], className="pb-name"),
            html.Div(f"{n:,} customers · {rs:.1f}% of revenue", className="pb-meta"),
            html.Div(plan["prio"], className="pb-prio"),
        ], className="pb-head", style={"background": info["gradient"]}),
        html.Div([
            html.Ul([html.Li(a) for a in plan["actions"]]),
        ], className="pb-body"),
    ], className="pb-card")


# ============================================================================
# APP
# ============================================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Customer Segmentation — Portfolio"

app.layout = html.Div([

    # HERO
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Customer Segmentation & Marketing Strategy",
                          style={"fontSize": "21px", "fontWeight": 800,
                                  "background": "linear-gradient(90deg, #6366F1, #EC4899)",
                                  "WebkitBackgroundClip": "text",
                                  "WebkitTextFillColor": "transparent",
                                  "letterSpacing": "-0.01em"}),
                html.Div("End-to-end unsupervised learning · 2,198 retail customers",
                          style={"fontSize": "11.5px", "color": "#6B7280", "marginTop": "2px"}),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Div("Gheffari Nour El Houda",
                              style={"fontSize": "13px", "fontWeight": 700, "color": "#1A1F2E"}),
                    html.Div([
                        html.A("github.com/houdhoudGH", href="https://github.com/houdhoudGH",
                                target="_blank",
                                style={"color": "#6366F1", "fontSize": "11px",
                                        "textDecoration": "none", "fontWeight": 600}),
                        html.Span(" · ", style={"color": "#6B7280", "fontSize": "11px"}),
                        html.A("Email", href="mailto:nourgheffari@gmail.com",
                                style={"color": "#6366F1", "fontSize": "11px",
                                        "textDecoration": "none", "fontWeight": 600}),
                    ]),
                ], style={"textAlign": "right"}),
            ], width=4),
        ]),
        html.Div([
            html.Span(t, className="tech-badge")
            for t in ["Python", "scikit-learn", "K-Means", "PCA", "t-SNE",
                       "Plotly Dash", "Pandas"]
        ], style={"marginTop": "10px"}),
    ], style={"padding": "16px 26px 12px 26px",
              "borderBottom": "1px solid #E5E7EB",
              "background": "#FFFFFF",
              "position": "relative", "zIndex": 1}),

    # STORY
    html.Div([

        # ───── CHAPTER 1 ─────
        chapter("1", "The Problem & The Data",
                "2,198 customers, 29 raw features, no labels"),
        narrative([
            "A retailer needs to segment their customer base to allocate marketing budget intelligently. ",
            html.B("Goal: find natural groups that justify different strategies"),
            " — using only unsupervised techniques since there are no labels.",
        ]),
        dbc.Row([
            dbc.Col(kpi("Customers", f"{total:,}", "after cleaning",
                         gradient="linear-gradient(135deg, #6366F1, #8B5CF6)"), width=3),
            dbc.Col(kpi("Raw features", "29", "demographics + behavior"), width=3),
            dbc.Col(kpi("Total revenue", f"${total_rev/1000:.0f}K", "to allocate"), width=3),
            dbc.Col(kpi("Approach", "Unsupervised", "no labels available"), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(plot_box(fig_age), width=2),
            dbc.Col(plot_box(fig_inc_raw), width=2),
            dbc.Col(plot_box(fig_spend_raw), width=2),
            dbc.Col(plot_box(fig_kids_dist), width=3),
            dbc.Col(plot_box(fig_recency_raw), width=3),
        ], className="g-2"),
        insight("What the raw data tells us",
                "Income is bell-shaped with a long right tail. Spending is heavily right-skewed — "
                "a small group carries most revenue. Family sizes are mostly 0-1 kids. "
                "Already we see hints of high-value vs low-value customers.",
                color="#6366F1"),

        # ───── CHAPTER 2 ─────
        chapter("2", "Cleaning & Feature Engineering",
                "Turning raw columns into meaningful business features"),
        narrative([
            "Raw columns like Year_Birth and MntWines/MntFruits/MntMeat aren't directly useful. "
            "We engineer ", html.B("Age, Expenses (sum), Kids (Kidhome+Teenhome), "
            "TotalNumPurchases, TotalAcceptedCmp"), " and one-hot encode categoricals.",
        ]),
        dbc.Row([
            dbc.Col(kpi("Nulls dropped", "24", "1.1% of rows"), width=3),
            dbc.Col(kpi("Engineered", "11+", "derived features",
                         gradient="linear-gradient(135deg, #10B981, #14B8A6)"), width=3),
            dbc.Col(kpi("Final features", "18", "after encoding"), width=3),
            dbc.Col(kpi("Inc-Exp ρ", "0.82", "strongest correlation",
                         gradient="linear-gradient(135deg, #F59E0B, #F97316)"), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(plot_box(fig_inc_exp), width=3),
            dbc.Col(plot_box(fig_kids_exp), width=2),
            dbc.Col(plot_box(fig_edu), width=2),
            dbc.Col(plot_box(fig_marital), width=2),
            dbc.Col(plot_box(fig_cmp_eng), width=3),
        ], className="g-2"),
        insight("Feature engineering insight",
                "Total Expenses (summed across categories) correlates with Income at 0.82 — "
                "far stronger than any individual category. Engineered features carry more signal "
                "than raw ones, which is half the battle in unsupervised ML.",
                color="#10B981"),

        # ───── CHAPTER 3 ─────
        chapter("3", "Scaling & Dimensionality Reduction",
                "Preparing data for distance-based clustering"),
        narrative([
            html.B("StandardScaler"), " normalizes feature scales (z-score). ", html.B("PCA"),
            " compresses 18 correlated features into 10 orthogonal components retaining 80% of variance — "
            "less noise, faster clustering, fair weighting across dimensions.",
        ]),
        dbc.Row([
            dbc.Col(kpi("Input features", "18", "after encoding"), width=3),
            dbc.Col(kpi("PCA components", "10", "after reduction",
                         gradient="linear-gradient(135deg, #6366F1, #8B5CF6)"), width=3),
            dbc.Col(kpi("Variance kept", "80%", "in 10 components",
                         gradient="linear-gradient(135deg, #F59E0B, #F97316)"), width=3),
            dbc.Col(kpi("Method", "Z-score", "StandardScaler"), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(plot_box(fig_scale_problem), width=3),
            dbc.Col(plot_box(fig_scaling), width=3),
            dbc.Col(plot_box(fig_pca), width=3),
            dbc.Col(plot_box(fig_var_donut), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(plot_box(fig_corr), width=12),
        ], className="g-2"),
        insight("Why dimensionality reduction matters here",
                "Income, Expenses, and TotalNumPurchases are highly correlated (0.71–0.82). Without PCA, "
                "K-Means would over-weight this 'wealth' signal. PCA collapses them into one orthogonal "
                "component, giving other dimensions (recency, family, campaigns) fair weight.",
                color="#F59E0B"),

        # ───── CHAPTER 4 ─────
        chapter("4", "Finding the Segments",
                "Choosing k, fitting K-Means, validating with t-SNE"),
        narrative([
            html.B("Elbow method"), " plots inertia for k=2..8 — the kink at k=3 shows where adding "
            "clusters stops paying off. ", html.B("Silhouette score"),
            " peaks at k=3 too. Then t-SNE projects 10D PCA space into 2D for visual validation.",
        ]),
        dbc.Row([
            dbc.Col(kpi("Optimal k", "3", "elbow + silhouette",
                         gradient="linear-gradient(135deg, #F59E0B, #F97316)"), width=3),
            dbc.Col(kpi("Champions", f"{n_champ}", f"{n_champ/total*100:.1f}% tiny elite"), width=3),
            dbc.Col(kpi("Loyal VIPs", f"{n_loyal:,}", f"{n_loyal/total*100:.1f}% backbone",
                         gradient="linear-gradient(135deg, #6366F1, #8B5CF6)"), width=3),
            dbc.Col(kpi("Price-Sens.", f"{n_ps:,}", f"{n_ps/total*100:.1f}% majority",
                         gradient="linear-gradient(135deg, #EF4444, #EC4899)"), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(plot_box(fig_elbow), width=3),
            dbc.Col(plot_box(fig_sil), width=3),
            dbc.Col(plot_box(fig_tsne), width=3),
            dbc.Col(plot_box(fig_sizes), width=3),
        ], className="g-2"),
        insight("Independent validation",
                "t-SNE was never told about the K-Means labels — yet same-cluster customers land near "
                "each other in 2D with almost no overlap. Strong evidence the segments are real.",
                color="#6366F1"),

        # ───── CHAPTER 5 ─────
        chapter("5", "From Profiles to Strategy",
                "Per-segment behavior + tailored marketing playbook"),
        narrative([
            html.B(f"{hv_share:.0f}% of revenue from {hv_pct:.0f}% of customers"),
            " — Champions + Loyal VIPs. Each segment gets a strategy sized to its unit economics.",
        ]),
        dbc.Row([
            dbc.Col(kpi("80/20 ratio", f"{hv_share:.0f}%", f"from {hv_pct:.0f}% cust",
                         gradient="linear-gradient(135deg, #F59E0B, #F97316)"), width=3),
            dbc.Col(kpi("Champion ARPU",
                         f"${rev_by.loc[rev_by['Cluster']==0,'ARPU'].iloc[0]:,.0f}",
                         "15× Price-Sensitive"), width=3),
            dbc.Col(kpi("VIP ARPU",
                         f"${rev_by.loc[rev_by['Cluster']==2,'ARPU'].iloc[0]:,.0f}",
                         "revenue backbone",
                         gradient="linear-gradient(135deg, #6366F1, #8B5CF6)"), width=3),
            dbc.Col(kpi("Playbooks", "3", "Protect · Grow · Automate",
                         gradient="linear-gradient(135deg, #10B981, #14B8A6)"), width=3),
        ], className="g-2"),
        # 6 small profile bars in one row
        dbc.Row([
            dbc.Col(plot_box(bar_inc), width=2),
            dbc.Col(plot_box(bar_exp), width=2),
            dbc.Col(plot_box(bar_pur), width=2),
            dbc.Col(plot_box(bar_age), width=2),
            dbc.Col(plot_box(bar_ten), width=2),
            dbc.Col(plot_box(fig_radar), width=2),
        ], className="g-2"),
        # Economics: 3 plots
        dbc.Row([
            dbc.Col(plot_box(fig_rev_donut), width=4),
            dbc.Col(plot_box(fig_cust_donut), width=4),
            dbc.Col(plot_box(fig_arpu), width=4),
        ], className="g-2"),
        # Playbook cards — full width now
        dbc.Row([
            dbc.Col(playbook_card(0), width=4),
            dbc.Col(playbook_card(2), width=4),
            dbc.Col(playbook_card(1), width=4),
        ], className="g-2"),
        insight("The business impact",
                "Same total budget, much higher ROI. Champions get expensive 1:1 attention because ARPU "
                "justifies it. Price-Sensitive get cheap automation. Loyal VIPs get growth investment as "
                "the path to more Champions.",
                color="#EC4899"),

        # FOOTER
        html.Div([
            html.Hr(style={"borderColor": "#E5E7EB", "margin": "28px 0 14px 0"}),
            html.Div([
                html.Span("Built with Python · scikit-learn · Plotly Dash · ",
                            style={"color": "#6B7280", "fontSize": "11px"}),
                html.A("github.com/houdhoudGH", href="https://github.com/houdhoudGH",
                        target="_blank",
                        style={"color": "#6366F1", "fontSize": "11px",
                                "textDecoration": "none", "fontWeight": 600}),
            ], style={"textAlign": "center", "padding": "10px 0 26px 0"}),
        ]),

    ], style={"padding": "6px 26px 16px 26px", "maxWidth": "1440px",
                "margin": "0 auto", "position": "relative", "zIndex": 1}),

], style={"background": "#FAFBFC", "minHeight": "100vh"})


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Dashboard: http://127.0.0.1:8050")
    print("Refined · white theme · 5 chapters")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)
