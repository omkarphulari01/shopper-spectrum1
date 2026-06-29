"""
Precompute everything the Streamlit dashboard needs and bundle it into a
single artifact, so the app loads instantly without recomputing from raw data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger("shopper_spectrum.models.dashboard")


def build_dashboard_data(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    X: np.ndarray,
    metrics: pd.DataFrame,
    sim_df: pd.DataFrame,
    code2name: dict,
    chosen_k: int,
    silhouette: float,
) -> dict:
    """Aggregate KPIs, time series, geo, PCA, heatmap and insights."""
    total_revenue = float(df["TotalPrice"].sum())
    n_invoices = int(df["InvoiceNo"].nunique())

    kpis = {
        "customers": int(rfm["CustomerID"].nunique()),
        "products": int(df["StockCode"].nunique()),
        "total_revenue": total_revenue,
        "transactions": n_invoices,
        "countries": int(df["Country"].nunique()),
        "top_country": df.groupby("Country")["TotalPrice"].sum().idxmax(),
        "avg_order_value": float(total_revenue / len(df)),
        "silhouette": float(silhouette),
        "optimal_k": int(chosen_k),
    }

    # Monthly revenue
    monthly = df.set_index("InvoiceDate")["TotalPrice"].resample("ME").sum()
    monthly_revenue = {
        "labels": [d.strftime("%Y-%m") for d in monthly.index],
        "values": [float(v) for v in monthly.values],
    }

    # Top products by quantity
    top_prod = (df.groupby("Description")["Quantity"].sum()
                  .sort_values(ascending=False).head(10))
    top_products = {"labels": top_prod.index.tolist(),
                    "values": [int(v) for v in top_prod.values]}

    # Country breakdowns (top 10, revenue & transactions)
    cr = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    ct = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False).head(10)
    country_revenue = {"labels": cr.index.tolist(), "values": [float(v) for v in cr.values]}
    country_transactions = {"labels": ct.index.tolist(), "values": [int(v) for v in ct.values]}

    # RFM distributions (capped at 99th pct for display)
    rfm_dist = {}
    for col in ["Recency", "Frequency", "Monetary"]:
        capped = rfm[col].clip(upper=rfm[col].quantile(0.99))
        rfm_dist[col] = [float(v) for v in capped.values]

    # Segment profiles + revenue
    seg_profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    seg_profile["Customers"] = rfm["Segment"].value_counts()
    seg_profile["Revenue"] = rfm.groupby("Segment")["Monetary"].sum()
    segment_table = seg_profile.reset_index().to_dict("records")

    # Normalized cluster-profile heatmap (0..1 per RFM column, by segment order)
    order = ["High-Value", "Regular", "Occasional", "At-Risk"]
    order = [s for s in order if s in seg_profile.index]
    norm = seg_profile.loc[order, ["Recency", "Frequency", "Monetary"]].copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - lo) / (hi - lo) if hi > lo else 0.0
    heatmap = {"segments": order,
               "columns": ["Recency", "Frequency", "Monetary"],
               "values": norm.values.tolist()}

    # PCA of the scaled RFM features for the 2D segment scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    pca_data = {
        "x": [float(v) for v in coords[:, 0]],
        "y": [float(v) for v in coords[:, 1]],
        "segment": rfm["Segment"].tolist(),
    }

    # Similarity matrix for the top 15 products
    top15 = (df.groupby("StockCode")["Quantity"].sum()
               .sort_values(ascending=False).index)
    top15 = [c for c in top15 if c in sim_df.index][:15]
    sim15 = sim_df.loc[top15, top15]
    similarity = {
        "labels": [str(code2name.get(c, c))[:24] for c in top15],
        "matrix": sim15.round(3).values.tolist(),
    }

    # Business insights
    hv = rfm[rfm["Segment"] == "High-Value"]
    occ = rfm[rfm["Segment"] == "Occasional"]
    atrisk = rfm[rfm["Segment"] == "At-Risk"]
    insights = {
        "hv_pct_customers": round(100 * len(hv) / len(rfm), 1),
        "hv_pct_revenue": round(100 * hv["Monetary"].sum() / rfm["Monetary"].sum(), 1),
        "occ_pct_customers": round(100 * len(occ) / len(rfm), 1),
        "atrisk_pct_customers": round(100 * len(atrisk) / len(rfm), 1),
        "avg_order_value": round(kpis["avg_order_value"], 2),
        "top_country": kpis["top_country"],
        "silhouette": round(silhouette, 4),
    }

    # Product list for the recommendation dropdown
    product_names = sorted({str(code2name.get(c, c)) for c in sim_df.index})

    # Elbow / silhouette data for KMeans
    km = metrics[metrics["algorithm"] == "kmeans"][["k", "inertia", "silhouette"]]
    elbow = {
        "k": [int(v) for v in km["k"].values],
        "inertia": [float(v) for v in km["inertia"].values],
        "silhouette": [float(v) for v in km["silhouette"].values],
    }

    logger.info("Dashboard data assembled")
    return {
        "kpis": kpis,
        "monthly_revenue": monthly_revenue,
        "top_products": top_products,
        "country_revenue": country_revenue,
        "country_transactions": country_transactions,
        "rfm_dist": rfm_dist,
        "segment_table": segment_table,
        "heatmap": heatmap,
        "pca": pca_data,
        "similarity": similarity,
        "insights": insights,
        "product_names": product_names,
        "elbow": elbow,
    }
