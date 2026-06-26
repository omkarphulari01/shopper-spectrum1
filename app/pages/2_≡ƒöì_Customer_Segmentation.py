"""Customer Segmentation module."""

import streamlit as st

from utils import SEGMENT_BLURB, SEGMENT_COLORS, load_artifacts, predict_segment

st.set_page_config(page_title="Customer Segmentation", page_icon="🔍", layout="wide")
art = load_artifacts()

st.title("🔍 Customer Segmentation")
st.write("Enter the customer's RFM values to predict which segment they belong to.")

c1, c2, c3 = st.columns(3)
with c1:
    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
with c2:
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
with c3:
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0, step=50.0)

if st.button("Predict Cluster", type="primary"):
    segment = predict_segment(art, recency, frequency, monetary)
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
