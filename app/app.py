"""
Shopper Spectrum — Streamlit web application.

Two modules:
  1. Product Recommendation  — enter a product name, get 5 similar products
  2. Customer Segmentation   — enter Recency/Frequency/Monetary, predict segment

Run from the project root:
    streamlit run app/app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")


# --------------------------------------------------------------------------
# Load artifacts (cached so they load once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    km = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_map = joblib.load(os.path.join(MODELS_DIR, "segment_labels.pkl"))
    rec = joblib.load(os.path.join(MODELS_DIR, "recommender.pkl"))
    return km, scaler, label_map, rec


try:
    km, scaler, label_map, rec = load_artifacts()
    NEIGHBORS = rec["neighbors"]
    CODE2NAME = rec["code2name"]
    NAME2CODE = rec["name2code"]
    ARTIFACTS_OK = True
except Exception as e:  # pragma: no cover
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)


SEGMENT_COLORS = {
    "High-Value": "#2E7D32",
    "Regular": "#1565C0",
    "Occasional": "#F9A825",
    "At-Risk": "#C62828",
}
SEGMENT_BLURB = {
    "High-Value": "Recent, frequent, big spenders. Reward and retain them.",
    "Regular": "Steady purchasers — nurture toward high-value.",
    "Occasional": "Rare, low-spend buyers — re-engage with offers.",
    "At-Risk": "Haven't purchased in a long time — win-back campaigns.",
}


# --------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------
def recommend(product_name, n=5):
    query = str(product_name).strip().lower()
    code = NAME2CODE.get(query)
    if code is None:
        matches = [name for name in NAME2CODE if query in name]
        if not matches:
            return None  # no match at all
        code = NAME2CODE[matches[0]]
    if code not in NEIGHBORS:
        return []
    return [
        {"product": CODE2NAME.get(c, c), "score": s}
        for c, s in NEIGHBORS[code][:n]
    ]


def predict_segment(recency, frequency, monetary):
    feats = np.array([[recency, np.log1p(frequency), np.log1p(monetary)]])
    scaled = scaler.transform(feats)
    cluster = int(km.predict(scaled)[0])
    return label_map.get(cluster, f"Cluster {cluster}")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendations in E-Commerce")

if not ARTIFACTS_OK:
    st.error(
        "Could not load model artifacts. Run `python src/train_pipeline.py` "
        f"from the project root first.\n\nDetails: {LOAD_ERROR}"
    )
    st.stop()

tab_rec, tab_seg = st.tabs(["🎯 Product Recommendation", "🔍 Customer Segmentation"])

# ---- Module 1: Product Recommendation ----
with tab_rec:
    st.subheader("Find similar products")
    st.write("Enter a product name to get 5 similar products (item-based collaborative filtering).")

    sample = list(CODE2NAME.values())[:8]
    with st.expander("Need ideas? Example product names"):
        st.write(", ".join(sample))

    product = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")
    if st.button("Get Recommendations", type="primary"):
        if not product.strip():
            st.warning("Please enter a product name.")
        else:
            results = recommend(product, n=5)
            if results is None:
                st.error("No matching product found. Try a different name or check spelling.")
            elif not results:
                st.info("That product has no similarity data available.")
            else:
                st.success(f"Top {len(results)} products similar to your input:")
                cols = st.columns(len(results))
                for col, item in zip(cols, results):
                    with col:
                        st.markdown(
                            f"<div style='border:1px solid #ddd;border-radius:10px;"
                            f"padding:14px;min-height:130px'>"
                            f"<b>{item['product'].title()}</b><br><br>"
                            f"<span style='color:#888'>similarity</span><br>"
                            f"<span style='font-size:1.4em;color:#1565C0'>{item['score']:.2f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

# ---- Module 2: Customer Segmentation ----
with tab_seg:
    st.subheader("Predict a customer's segment")
    st.write("Enter the customer's RFM values to predict which segment they belong to.")

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
    with c3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0, step=50.0)

    if st.button("Predict Cluster", type="primary"):
        segment = predict_segment(recency, frequency, monetary)
        color = SEGMENT_COLORS.get(segment, "#444")
        st.markdown(
            f"<div style='background:{color};color:white;border-radius:12px;"
            f"padding:24px;text-align:center;margin-top:10px'>"
            f"<div style='font-size:0.9em;opacity:0.85'>Predicted Segment</div>"
            f"<div style='font-size:2em;font-weight:700'>{segment}</div>"
            f"<div style='margin-top:8px'>{SEGMENT_BLURB.get(segment, '')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption("Built with KMeans (RFM segmentation) + item-based collaborative filtering.")
