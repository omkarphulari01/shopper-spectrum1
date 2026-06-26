"""Product Recommendation module."""

import streamlit as st

from utils import load_artifacts, recommend_products, sample_product_names

st.set_page_config(page_title="Product Recommendation", page_icon="🎯", layout="wide")
art = load_artifacts()

st.title("🎯 Product Recommendation")
st.write("Enter a product name to get 5 similar products (item-based collaborative filtering).")

with st.expander("Need ideas? Example product names"):
    st.write(", ".join(sample_product_names(art)))

product = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

if st.button("Get Recommendations", type="primary"):
    if not product.strip():
        st.warning("Please enter a product name.")
    else:
        results = recommend_products(art, product, n=5)
        if results is None:
            st.error("No matching product found. Try a different name or spelling.")
        elif not results:
            st.info("That product has no similarity data available.")
        else:
            st.success(f"Top {len(results)} products similar to your input:")
            cols = st.columns(len(results))
            for col, item in zip(cols, results):
                with col:
                    st.markdown(
                        f"<div style='border:1px solid #ddd;border-radius:10px;"
                        f"padding:14px;min-height:140px'>"
                        f"<b>{item['product'].title()}</b><br><br>"
                        f"<span style='color:#888'>similarity</span><br>"
                        f"<span style='font-size:1.4em;color:#1565C0'>{item['score']:.2f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
