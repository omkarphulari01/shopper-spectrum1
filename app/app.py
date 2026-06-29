"""
Shopper Spectrum — full analytics dashboard.

A dark-themed, multi-section Streamlit app powered by precomputed dashboard
data plus the trained models. Sections:

    Executive Dashboard · Sales Analytics · Country Analysis · RFM Analysis ·
    Elbow Method · Customer Segmentation · Similarity Matrix ·
    Product Recommendation · Customer Prediction · Business Insights

Run from the project root:
    streamlit run app/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from utils import (
    SEGMENT_BLURB,
    SEGMENT_COLORS,
    SEGMENT_EMOJI,
    load_artifacts,
    predict_segment,
    recommend_products,
)

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .insight-box { background:#161B2E; border-left:4px solid #3B82F6;
                     border-radius:8px; padding:14px 16px; margin:8px 0; }
      .finding { background:#161B2E; border-left:4px solid #F59E0B;
                 border-radius:8px; padding:12px 14px; margin:6px 0;
                 font-size:0.92rem; }
      .seg-desc { background:#161B2E; border-radius:8px; padding:12px 14px;
                  margin:6px 0; font-size:0.9rem; }
      h1, h2, h3 { font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161B2E",
    font=dict(color="#E5E7EB"),
    margin=dict(l=10, r=10, t=50, b=10),
)

try:
    art = load_artifacts()
    D = art["dashboard"]
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
    st.caption("Navigation")
    page = option_menu(
        menu_title=None,
        options=[
            "Executive Dashboard", "Sales Analytics", "Country Analysis",
            "RFM Analysis", "Elbow Method", "Customer Segmentation",
            "Similarity Matrix", "Product Recommendation",
            "Customer Prediction", "Business Insights",
        ],
        icons=[
            "speedometer2", "graph-up", "globe", "rulers", "diagram-3",
            "people", "grid-3x3", "bullseye", "magic", "lightbulb",
        ],
        default_index=0,
        styles={
            "container": {"padding": "4px", "background-color": "#0E1117"},
            "icon": {"color": "#9CA3AF", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "2px",
                "color": "#E5E7EB", "--hover-color": "#1A1F2E",
            },
            "nav-link-selected": {"background-color": "#FF4B6E", "color": "white"},
        },
    )


def section_header(emoji, title, subtitle):
    st.markdown(f"# {emoji} {title}")
    st.caption(subtitle)
    st.divider()


def fmt_money(v):
    if v >= 1e6:
        return f"£{v/1e6:.0f}M" if v >= 1e7 else f"£{v/1e6:.2f}M"
    if v >= 1e3:
        return f"£{v/1e3:.0f}K"
    return f"£{v:.0f}"


# ===========================================================================
# 1. EXECUTIVE DASHBOARD
# ===========================================================================
def render_executive():
    section_header("📊", "Executive Dashboard", "High-level overview of business performance")
    k = D["kpis"]
    c = st.columns(5)
    c[0].metric("👥 Customers", f"{k['customers']:,}")
    c[1].metric("📦 Products", f"{k['products']:,}")
    c[2].metric("💰 Total Revenue", fmt_money(k["total_revenue"]))
    c[3].metric("🧾 Transactions", f"{k['transactions']:,}")
    c[4].metric("🌍 Countries", f"{k['countries']}")

    c = st.columns(3)
    c[0].metric("🏆 Top Country", k["top_country"])
    c[1].metric("⭐ Avg Order Value", f"£{k['avg_order_value']:.2f}")
    c[2].metric("🎯 Silhouette Score", f"{k['silhouette']:.4f}")
    st.divider()

    st.markdown("### 📈 Monthly Revenue Trend")
    _revenue_trend()
    st.markdown("### 🏆 Top Products")
    _top_products()


def _revenue_trend():
    m = D["monthly_revenue"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["labels"], y=m["values"], mode="lines+markers",
        line=dict(color="#22D3EE", width=2), fill="tozeroy",
        fillcolor="rgba(34,211,238,0.18)"))
    fig.update_layout(title="Monthly Revenue Trend", height=380, **PLOTLY_LAYOUT)
    fig.update_yaxes(title="Revenue (£)", gridcolor="#2A3142")
    fig.update_xaxes(gridcolor="#2A3142")
    st.plotly_chart(fig, use_container_width=True)


def _top_products():
    t = D["top_products"]
    n = len(t["labels"])
    colors = px.colors.sample_colorscale("YlGn", [i / max(n - 1, 1) for i in range(n)])
    fig = go.Figure(go.Bar(
        x=t["values"][::-1], y=t["labels"][::-1], orientation="h",
        marker=dict(color=colors[::-1])))
    fig.update_layout(title="Top 10 Best-Selling Products", height=420, **PLOTLY_LAYOUT)
    fig.update_xaxes(title="Total Quantity Sold", gridcolor="#2A3142")
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 2. SALES ANALYTICS
# ===========================================================================
def render_sales():
    section_header("📈", "Sales Analytics", "Revenue trends and product performance")
    st.markdown("### 🗓️ Monthly Revenue Trend")
    _revenue_trend()
    st.markdown("### 🏆 Top 10 Best-Selling Products")
    _top_products()

    st.markdown("### 📊 Key Sales Metrics")
    m = D["monthly_revenue"]
    s = pd.Series(m["values"], index=m["labels"])
    c = st.columns(3)
    c[0].metric("Peak Revenue Month", s.idxmax(), fmt_money(s.max()))
    c[1].metric("Lowest Revenue Month", s.idxmin(), fmt_money(s.min()))
    c[2].metric("Avg Monthly Revenue", fmt_money(s.mean()))


# ===========================================================================
# 3. COUNTRY ANALYSIS
# ===========================================================================
def render_country():
    section_header("🌍", "Country Analysis", "Geographic distribution of sales and customers")
    cr, ct = D["country_revenue"], D["country_transactions"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=cr["values"][::-1], y=cr["labels"][::-1],
                               orientation="h", marker_color="#FBBF24"))
        fig.update_layout(title="Revenue by Country (£)", height=400, **PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(x=ct["values"][::-1], y=ct["labels"][::-1],
                               orientation="h", marker_color="#C026D3"))
        fig.update_layout(title="Transactions by Country", height=400, **PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<div class='insight-box'>💡 <b>Top Country:</b> {D['kpis']['top_country']} "
        f"generates the highest revenue.<br>🌐 <b>Global Reach:</b> "
        f"{D['kpis']['countries']} countries served worldwide.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# 4. RFM ANALYSIS
# ===========================================================================
def render_rfm():
    section_header("📐", "RFM Analysis",
                   "Recency · Frequency · Monetary — the foundation of customer segmentation")
    c = st.columns(3)
    c[0].markdown("📅 **Recency** Days since last purchase. Lower = more recent customer.")
    c[1].markdown("📦 **Frequency** Number of orders placed. Higher = more loyal customer.")
    c[2].markdown("💰 **Monetary** Total amount spent. Higher = more valuable customer.")
    st.divider()

    st.markdown("### 📊 RFM Distribution Charts")
    rd = D["rfm_dist"]
    cols = st.columns(3)
    palette = {"Recency": "#EF4444", "Frequency": "#3B82F6", "Monetary": "#22C55E"}
    for col, name in zip(cols, ["Recency", "Frequency", "Monetary"]):
        fig = go.Figure(go.Histogram(x=rd[name], marker_color=palette[name], nbinsx=40))
        fig.update_layout(title=name, height=300, **PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor="#2A3142"); fig.update_yaxes(gridcolor="#2A3142")
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Cluster Profile Heatmap")
    _heatmap()

    st.markdown("### 📋 Segment RFM Averages")
    df = pd.DataFrame(D["segment_table"]).set_index("Segment")
    show = df[["Recency", "Frequency", "Monetary"]].round(2)
    show.columns = ["Recency (days)", "Frequency (orders)", "Monetary (£)"]
    st.dataframe(show, use_container_width=True)


def _heatmap():
    h = D["heatmap"]
    fig = go.Figure(go.Heatmap(
        z=h["values"], x=h["columns"], y=h["segments"],
        colorscale="RdYlGn", text=[[f"{v:.2f}" for v in row] for row in h["values"]],
        texttemplate="%{text}", showscale=True))
    fig.update_layout(title="Cluster Profile Heatmap (Normalized RFM)", height=360,
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 5. ELBOW METHOD
# ===========================================================================
def render_elbow():
    section_header("📉", "Elbow Method", "Finding the optimal number of clusters (K)")
    st.markdown(
        "The **Elbow Method** helps decide how many customer groups (K) to create. "
        "We look for the point where inertia stops dropping sharply — the *elbow*. "
        "The **Silhouette Score** confirms the best K by measuring how well-separated "
        "the clusters are."
    )
    e = D["elbow"]
    k_opt = D["kpis"]["optimal_k"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scatter(x=e["k"], y=e["inertia"], mode="lines+markers",
                                   line=dict(color="#22D3EE")))
        fig.add_vline(x=k_opt, line_dash="dash", line_color="#EF4444")
        fig.update_layout(title="Elbow Curve", height=340, **PLOTLY_LAYOUT)
        fig.update_xaxes(title="Number of Clusters (K)", gridcolor="#2A3142")
        fig.update_yaxes(title="Inertia", gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Scatter(x=e["k"], y=e["silhouette"], mode="lines+markers",
                                   line=dict(color="#EF4444")))
        fig.add_vline(x=k_opt, line_dash="dash", line_color="#22D3EE")
        fig.update_layout(title="Silhouette Score", height=340, **PLOTLY_LAYOUT)
        fig.update_xaxes(title="Number of Clusters (K)", gridcolor="#2A3142")
        fig.update_yaxes(title="Silhouette Score", gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)

    inertia_at_k = e["inertia"][e["k"].index(k_opt)] if k_opt in e["k"] else None
    c = st.columns(3)
    c[0].metric("Optimal K", k_opt)
    if inertia_at_k:
        c[1].metric("Inertia at K", f"{inertia_at_k:,.0f}")
    c[2].metric("Silhouette Score", f"{D['kpis']['silhouette']:.4f}")
    st.markdown(
        f"<div class='insight-box'>✅ <b>K={k_opt} is optimal</b> — beyond this the "
        f"inertia reduction becomes minimal (diminishing returns).<br>"
        f"✅ <b>Silhouette Score {D['kpis']['silhouette']:.2f}</b> indicates "
        f"well-separated, meaningful clusters.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# 6. CUSTOMER SEGMENTATION
# ===========================================================================
def render_segmentation():
    section_header("👥", "Customer Segmentation",
                   f"KMeans Clustering on RFM features — K={D['kpis']['optimal_k']}")
    table = {r["Segment"]: r for r in D["segment_table"]}
    order = [s for s in ["High-Value", "Regular", "Occasional", "At-Risk"] if s in table]
    cols = st.columns(len(order))
    for col, seg in zip(cols, order):
        col.metric(f"{SEGMENT_EMOJI[seg]} {seg}", f"{int(table[seg]['Customers']):,}",
                   fmt_money(table[seg]["Revenue"]) + " revenue")
    st.divider()

    st.markdown("### 🔵 PCA Cluster Visualization")
    _pca_scatter()
    st.markdown("### 📊 Segment Distribution & Revenue")
    _segment_bars(table, order)
    st.markdown("### 🔥 Cluster Profile Heatmap")
    _heatmap()

    st.markdown("### 💡 Segment Descriptions")
    c = st.columns(2)
    for i, seg in enumerate(order):
        c[i % 2].markdown(
            f"<div class='seg-desc'>{SEGMENT_EMOJI[seg]} <b>{seg}</b> "
            f"{SEGMENT_BLURB[seg]}</div>", unsafe_allow_html=True)


def _pca_scatter():
    p = D["pca"]
    df = pd.DataFrame({"x": p["x"], "y": p["y"], "Segment": p["segment"]})
    fig = px.scatter(df, x="x", y="y", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, opacity=0.6,
                     labels={"x": "Principal Component 1", "y": "Principal Component 2"})
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(title="Customer Segments — PCA Visualization", height=460,
                      **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#2A3142"); fig.update_yaxes(gridcolor="#2A3142")
    st.plotly_chart(fig, use_container_width=True)


def _segment_bars(table, order):
    col1, col2 = st.columns(2)
    sizes = [int(table[s]["Customers"]) for s in order]
    revs = [table[s]["Revenue"] / 1000 for s in order]
    colors = [SEGMENT_COLORS[s] for s in order]
    with col1:
        fig = go.Figure(go.Bar(x=order, y=sizes, marker_color=colors))
        fig.update_layout(title="Customers per Segment", height=340, **PLOTLY_LAYOUT)
        fig.update_yaxes(gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(x=order, y=revs, marker_color=colors))
        fig.update_layout(title="Revenue per Segment (£K)", height=340, **PLOTLY_LAYOUT)
        fig.update_yaxes(gridcolor="#2A3142")
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 7. SIMILARITY MATRIX
# ===========================================================================
def render_similarity():
    section_header("🔗", "Product Similarity Matrix",
                   "Cosine similarity between top 15 products")
    st.markdown(
        "This heatmap shows how similar products are to each other based on "
        "**co-purchase patterns**. A score close to **1.0** means customers who buy "
        "one product also tend to buy the other."
    )
    s = D["similarity"]
    fig = go.Figure(go.Heatmap(
        z=s["matrix"], x=s["labels"], y=s["labels"], colorscale="Blues",
        showscale=True))
    fig.update_layout(title="Product Similarity Matrix (Top 15)", height=620,
                      **PLOTLY_LAYOUT)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='insight-box'>🔵 <b>Dark blue</b> = very similar products "
        "(frequently bought together)<br>⚪ <b>Light</b> = less similar products<br>"
        "This matrix powers the product recommendation engine.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# 8. PRODUCT RECOMMENDATION
# ===========================================================================
def render_recommendation():
    section_header("🎯", "Product Recommendation System",
                   "Item-based Collaborative Filtering · Cosine Similarity")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select a Product:", [""] + D["product_names"][:1000])
    with col2:
        n = st.selectbox("# Recommendations:", [5, 10, 15], index=0)
    typed = st.text_input("Or type a product name (partial search supported):")

    if st.button("🎁 Get Recommendations", type="primary"):
        query = typed.strip() or selected
        if not query:
            st.warning("Select or type a product name.")
        else:
            results = recommend_products(art, query, n=n)
            if results is None:
                st.error("No matching product found.")
            elif not results:
                st.info("No similarity data for that product.")
            else:
                st.success(f"Top {len(results)} products similar to your choice:")
                for i, item in enumerate(results, 1):
                    st.markdown(
                        f"<div class='seg-desc'><b>{i}. {item['product'].title()}</b>"
                        f" &nbsp; <span style='color:#22D3EE'>"
                        f"similarity {item['score']:.2f}</span></div>",
                        unsafe_allow_html=True)


# ===========================================================================
# 9. CUSTOMER PREDICTION
# ===========================================================================
def render_prediction():
    section_header("🔮", "Customer Segment Prediction",
                   "Enter RFM values to predict which segment a customer belongs to")
    c = st.columns(3)
    recency = c[0].number_input("📅 Recency (days since last purchase)", 0, value=30, step=1)
    frequency = c[1].number_input("📦 Frequency (number of orders)", 1, value=5, step=1)
    monetary = c[2].number_input("💰 Monetary (total spend £)", 0.0, value=500.0, step=50.0)
    c[0].caption("Lower = more recent = better")
    c[1].caption("Higher = more loyal")
    c[2].caption("Higher = more valuable")

    if st.button("🔮 Predict Segment", type="primary"):
        seg = predict_segment(art, recency, frequency, monetary)
        color = SEGMENT_COLORS.get(seg, "#444")
        st.markdown(
            f"<div style='background:{color};color:white;border-radius:12px;"
            f"padding:24px;text-align:center;margin-top:14px'>"
            f"<div style='font-size:0.9em;opacity:0.85'>Predicted Segment</div>"
            f"<div style='font-size:2em;font-weight:800'>{SEGMENT_EMOJI.get(seg,'')} {seg}</div>"
            f"<div style='margin-top:6px'>{SEGMENT_BLURB.get(seg,'')}</div></div>",
            unsafe_allow_html=True)


# ===========================================================================
# 10. BUSINESS INSIGHTS
# ===========================================================================
def render_insights():
    section_header("💡", "Business Insights", "Key findings and actionable recommendations")
    ins = D["insights"]
    st.markdown("## 🔑 Key Findings")
    findings = [
        f"⭐ <b>High-Value customers</b> are only {ins['hv_pct_customers']}% of total "
        f"customers but contribute {ins['hv_pct_revenue']}% of revenue.",
        f"🔔 {ins['occ_pct_customers']}% customers are Occasional buyers — a huge "
        f"opportunity for conversion with the right campaigns.",
        f"⚠️ {ins['atrisk_pct_customers']}% of customers are At-Risk — targeted "
        f"win-back campaigns can recover significant revenue.",
        f"⭐ Average order value is £{ins['avg_order_value']} — bundle offers can "
        f"increase this significantly.",
        f"🌍 <b>{ins['top_country']}</b> is the top revenue-generating country.",
        f"🎯 Clustering Silhouette Score: <b>{ins['silhouette']}</b> — indicates "
        f"well-separated, meaningful customer groups.",
    ]
    cols = st.columns(2)
    for i, f in enumerate(findings):
        cols[i % 2].markdown(f"<div class='finding'>{f}</div>", unsafe_allow_html=True)

    st.markdown("## 📋 Recommended Actions by Segment")
    table = {r["Segment"]: r for r in D["segment_table"]}
    order = [s for s in ["High-Value", "Regular", "Occasional", "At-Risk"] if s in table]
    actions = {
        "High-Value": ("VIP rewards + exclusive access", "🔴 High"),
        "Regular": ("Bundle deals + upsell", "🟡 Medium"),
        "Occasional": ("Seasonal promos + reminders", "🟢 Normal"),
        "At-Risk": ("Win-back emails + discounts", "🔴 Urgent"),
    }
    rows = []
    for s in order:
        rows.append({
            "Segment": f"{SEGMENT_EMOJI[s]} {s}",
            "Size": int(table[s]["Customers"]),
            "Revenue": fmt_money(table[s]["Revenue"]),
            "Action": actions[s][0],
            "Priority": actions[s][1],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===========================================================================
# Router
# ===========================================================================
ROUTES = {
    "Executive Dashboard": render_executive,
    "Sales Analytics": render_sales,
    "Country Analysis": render_country,
    "RFM Analysis": render_rfm,
    "Elbow Method": render_elbow,
    "Customer Segmentation": render_segmentation,
    "Similarity Matrix": render_similarity,
    "Product Recommendation": render_recommendation,
    "Customer Prediction": render_prediction,
    "Business Insights": render_insights,
}
ROUTES[page]()
