"""
Shopper Spectrum — Streamlit dashboard.

A light-themed single-page app with a custom sidebar (Home / Clustering /
Recommendation), powered by streamlit-option-menu.

Run from the project root:
    streamlit run app/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from utils import (
    FIG_DIR,
    REPORTS_DIR,
    SEGMENT_BLURB,
    SEGMENT_COLORS,
    load_artifacts,
    predict_segment,
    recommend_products,
    sample_product_names,
)

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------------------
# Light, clean styling to match the dashboard mock-up
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1100px; }
      h1, h2, h3 { color: #1F2A44; font-weight: 800; }
      .ss-label { color:#7A869A; font-size:0.85rem; margin-bottom:2px; }
      .rec-item { color:#1F4E8C; font-size:1.02rem; padding:6px 0;
                  border-bottom:1px solid #EEF1F5; }
      .metric-card { background:#F4F6F9; border-radius:12px; padding:18px;
                     text-align:center; }
      .metric-num { font-size:1.8rem; font-weight:800; color:#1F2A44; }
      .metric-cap { color:#7A869A; font-size:0.85rem; }
      div.stButton > button {
          border:1.5px solid #FF4B4B; color:#FF4B4B; background:white;
          border-radius:8px; font-weight:600; padding:4px 18px;
      }
      div.stButton > button:hover { background:#FF4B4B; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load artifacts (graceful failure if pipeline hasn't run)
# ---------------------------------------------------------------------------
try:
    art = load_artifacts()
except Exception as e:
    st.error(
        "Model artifacts not found. Train first:\n\n"
        "```bash\nshopper-spectrum train\n```\n\n"
        f"Details: {e}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛒 Shopper Spectrum")
    page = option_menu(
        menu_title=None,
        options=["Home", "Clustering", "Recommendation"],
        icons=["house", "diagram-3", "list-task"],
        default_index=0,
        styles={
            "container": {"padding": "4px", "background-color": "#FFFFFF"},
            "icon": {"color": "#6B7280", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px", "text-align": "left", "margin": "4px",
                "color": "#1F2A44", "--hover-color": "#F4F6F9",
            },
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "white"},
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _img(col, name, caption):
    p = Path(FIG_DIR) / name
    if p.exists():
        col.image(str(p), caption=caption, use_container_width=True)


def _load_summary():
    p = Path(REPORTS_DIR) / "run_summary.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
def render_home():
    st.title("Shopper Spectrum")
    st.caption("Customer Segmentation & Product Recommendations in E-Commerce")

    summary = _load_summary()
    if summary:
        cols = st.columns(4)
        cards = [
            (f"{summary['rows_clean']:,}", "Clean transactions"),
            (f"{summary['customers']:,}", "Customers"),
            (f"{summary['products_in_recommender']:,}", "Products"),
            (f"{summary['k']}", "Segments"),
        ]
        for col, (num, cap) in zip(cols, cards):
            col.markdown(
                f"<div class='metric-card'><div class='metric-num'>{num}</div>"
                f"<div class='metric-cap'>{cap}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.subheader("Customer segments")
    profiles = Path(REPORTS_DIR) / "segment_profiles.csv"
    if profiles.exists():
        st.dataframe(pd.read_csv(profiles), use_container_width=True, hide_index=True)

    st.markdown(
        "Use the sidebar to **predict a customer's segment** or **find similar products**."
    )


# ---------------------------------------------------------------------------
# CLUSTERING (customer segmentation)
# ---------------------------------------------------------------------------
def render_clustering():
    st.title("Customer Segmentation")
    st.markdown("<div class='ss-label'>Enter the customer's RFM values</div>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    frequency = c2.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
    monetary = c3.number_input("Monetary (total spend)", min_value=0.0, value=1000.0, step=50.0)

    if st.button("Predict Cluster"):
        segment = predict_segment(art, recency, frequency, monetary)
        color = SEGMENT_COLORS.get(segment, "#444")
        st.markdown(
            f"<div style='background:{color};color:white;border-radius:12px;"
            f"padding:22px;text-align:center;margin-top:14px'>"
            f"<div style='font-size:0.9em;opacity:0.85'>Predicted Segment</div>"
            f"<div style='font-size:2em;font-weight:800'>{segment}</div>"
            f"<div style='margin-top:6px'>{SEGMENT_BLURB.get(segment, '')}</div></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Clustering insights"):
        t1, t2 = st.columns(2)
        _img(t1, "elbow_silhouette.png", "Elbow & silhouette")
        _img(t2, "algorithm_comparison.png", "Algorithm comparison")
        _img(t1, "cluster_scatter.png", "Segments: Recency vs Monetary")
        _img(t2, "cluster_scatter_3d.png", "3D RFM segments")


# ---------------------------------------------------------------------------
# RECOMMENDATION  (matches the mock-up)
# ---------------------------------------------------------------------------
def render_recommendation():
    st.title("Product Recommender")
    st.markdown("<div class='ss-label'>Enter Product Name</div>", unsafe_allow_html=True)

    product = st.text_input(
        "Enter Product Name", label_visibility="collapsed",
        placeholder="e.g. GREEN VINTAGE SPOT BEAKER",
    )

    with st.expander("Need ideas? Example product names"):
        st.write(", ".join(sample_product_names(art)))

    if st.button("Recommend"):
        if not product.strip():
            st.warning("Please enter a product name.")
        else:
            results = recommend_products(art, product, n=5)
            if results is None:
                st.error("No matching product found. Try a different name or spelling.")
            elif not results:
                st.info("That product has no similarity data available.")
            else:
                st.markdown("**Recommended Products:**")
                for item in results:
                    st.markdown(
                        f"<div class='rec-item'>{item['product'].upper()}</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Home":
    render_home()
elif page == "Clustering":
    render_clustering()
else:
    render_recommendation()
