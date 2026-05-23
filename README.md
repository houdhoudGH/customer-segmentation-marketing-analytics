# Customer Segmentation Dashboard — Plotly Dash

Dark-theme BI-style dashboard with 4 pages, dense plots, screenshot-ready.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050 in your browser.

## Folder structure required

```
dash_app/
├── app.py
├── requirements.txt
├── assets/
│   └── custom.css         # dark theme styles
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   └── feature_columns.json
└── data/
    ├── clustered_data.parquet
    ├── cluster_profiles.parquet
    └── tsne_coords.parquet
```

## To screenshot for portfolio

1. Run the app
2. Open in Chrome at full-screen (F11)
3. Click each tab (Overview, Exploration, Clusters, Strategy)
4. Press F12 → Ctrl+Shift+P → "Capture full size screenshot"
5. You now have 4 dashboard images for your CV/LinkedIn/GitHub
