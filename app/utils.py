"""Shared helpers for the Streamlit app: artifact loading and prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

SEGMENT_COLORS = {
    "High-Value": "#22C55E",
    "Regular": "#3B82F6",
    "Occasional": "#F59E0B",
    "At-Risk": "#EF4444",
}
SEGMENT_EMOJI = {
    "High-Value": "⭐",
    "Regular": "✅",
    "Occasional": "🔔",
    "At-Risk": "⚠️",
}
SEGMENT_BLURB = {
    "High-Value": "Recent buyers, purchase often, spend the most. Strategy: VIP rewards, exclusive offers, loyalty programs.",
    "Regular": "Consistent buyers with moderate spend. Strategy: Upsell, loyalty perks, cross-sell.",
    "Occasional": "Buy rarely, low spend. Strategy: Re-engagement emails, seasonal promotions.",
    "At-Risk": "Long time since last purchase. Strategy: Win-back campaigns, special discounts.",
}


@st.cache_resource
def load_artifacts():
    """Load all model artifacts once per session."""
    return {
        "model": joblib.load(MODELS_DIR / "cluster_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "labels": joblib.load(MODELS_DIR / "segment_labels.pkl"),
        "feature_config": joblib.load(MODELS_DIR / "feature_config.pkl"),
        "rec": joblib.load(MODELS_DIR / "recommender.pkl"),
        "dashboard": joblib.load(MODELS_DIR / "dashboard_data.pkl"),
    }


def predict_segment(art, recency, frequency, monetary) -> str:
    """Scale RFM input and predict the segment label."""
    fc = art["feature_config"]
    vals = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}
    row = []
    for col in fc["rfm_cols"]:
        v = vals[col]
        if col in fc["log_transform"]:
            v = np.log1p(v)
        row.append(v)
    scaled = art["scaler"].transform([row])
    cluster = int(np.asarray(art["model"].predict(scaled)).ravel()[0])
    return art["labels"].get(cluster, f"Cluster {cluster}")


def recommend_products(art, product_name, n=5):
    """Top-N similar products via the compact neighbour dict."""
    rec = art["rec"]
    neighbors, code2name, name2code = rec["neighbors"], rec["code2name"], rec["name2code"]
    query = str(product_name).strip().lower()
    code = name2code.get(query)
    if code is None:
        matches = [name for name in name2code if query in name]
        if not matches:
            return None
        code = name2code[matches[0]]
    if code not in neighbors:
        return []
    return [{"product": code2name.get(c, c), "score": s}
            for c, s in neighbors[code][:n]]


def sample_product_names(art, k=8):
    return list(art["rec"]["code2name"].values())[:k]
