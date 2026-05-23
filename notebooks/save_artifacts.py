# ============================================================
# RUN THIS AS THE LAST CELL OF YOUR NOTEBOOK
# ============================================================
# Saves all the artifacts the Streamlit app needs:
#   - scaler.pkl, pca.pkl, kmeans.pkl (the pipeline)
#   - cluster_profiles.parquet (median/mean stats per cluster)
#   - clustered_data.parquet (full data with cluster labels)
#   - tsne_coords.parquet (precomputed 2D coords for the map)
#   - feature_columns.json (exact column order the scaler expects)
# ============================================================

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# ---- ADJUST THESE NAMES TO MATCH YOUR NOTEBOOK VARIABLES ----
# Your trained objects:
SCALER = scaler          # the fitted StandardScaler
PCA_MODEL = pca          # the fitted PCA (10 components)
KMEANS = kmeans          # the fitted KMeans (k=3)

# Your dataframes:
X_SCALED = X_scaled      # scaled feature matrix before PCA (numpy or DataFrame)
X_PCA = X_pca            # PCA-transformed matrix (n_samples, 10)
FULL_DF = df             # cleaned df WITH engineered features (Age, Expenses, Kids, etc.)
                         # BEFORE encoding — we want human-readable columns for profiling
FEATURE_DF = X           # the exact DataFrame passed to scaler.fit (post-encoding)
# -------------------------------------------------------------

OUT = Path("../models")
DATA_OUT = Path("../data")
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

# 1. Pickle the pipeline
with open(OUT / "scaler.pkl", "wb") as f:
    pickle.dump(SCALER, f)
with open(OUT / "pca.pkl", "wb") as f:
    pickle.dump(PCA_MODEL, f)
with open(OUT / "kmeans.pkl", "wb") as f:
    pickle.dump(KMEANS, f)

# 2. Save the exact feature order the scaler expects (CRITICAL for prediction)
feature_columns = list(FEATURE_DF.columns)
with open(OUT / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

# 3. Attach cluster labels to the human-readable df and save
labels = KMEANS.predict(X_PCA)
clustered = FULL_DF.copy()
clustered["Cluster"] = labels
clustered.to_parquet(DATA_OUT / "clustered_data.parquet", index=False)

# 4. Cluster profiles (medians + counts) for the comparison page
profile_cols = [
    "Age", "Income", "Days_is_client", "Expenses",
    "TotalNumPurchases", "TotalAcceptedCmp", "Kids", "Recency",
]
profile_cols = [c for c in profile_cols if c in clustered.columns]
profiles = clustered.groupby("Cluster")[profile_cols].median().round(2)
profiles["count"] = clustered.groupby("Cluster").size()
profiles.to_parquet(DATA_OUT / "cluster_profiles.parquet")

# 5. Precompute t-SNE coordinates (so the app doesn't recompute on every load)
print("Computing t-SNE (60-90s)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
tsne_xy = tsne.fit_transform(X_PCA)
tsne_df = pd.DataFrame(tsne_xy, columns=["tsne_x", "tsne_y"])
tsne_df["Cluster"] = labels
tsne_df.to_parquet(DATA_OUT / "tsne_coords.parquet", index=False)

print("\nAll artifacts saved:")
for p in sorted(list(OUT.glob("*")) + list(DATA_OUT.glob("*"))):
    print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
