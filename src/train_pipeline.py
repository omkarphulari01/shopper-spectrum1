"""
End-to-end pipeline: clean data -> RFM -> choose k -> cluster -> label
segments -> build recommender -> save all artifacts and EDA figures.

Run from the project root:
    python src/train_pipeline.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_clean
from rfm import build_rfm
from clustering import scale_rfm, evaluate_k, fit_kmeans, label_segments
from recommender import build_similarity, build_top_neighbors

DATA_PATH = os.environ.get("DATA_PATH", "data/online_retail.csv")
MODELS_DIR = "models"
FIG_DIR = "reports/figures"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns_ok = True
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    sns_ok = False


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    print("1/6  Loading and cleaning data ...")
    df = load_and_clean(DATA_PATH)
    print(f"     {len(df):,} clean rows, {df['CustomerID'].nunique():,} customers")

    # ---------------- EDA figures ----------------
    print("2/6  Generating EDA figures ...")

    # Top 10 countries by transaction volume (excluding UK for readability too)
    top_countries = df["Country"].value_counts().head(10)
    plt.figure(figsize=(9, 5))
    top_countries.iloc[::-1].plot(kind="barh", color="#4C72B0")
    plt.title("Top 10 Countries by Transaction Volume")
    plt.xlabel("Number of line items")
    savefig("top_countries.png")

    # Top 10 selling products by quantity
    top_products = (
        df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    )
    plt.figure(figsize=(9, 5))
    top_products.iloc[::-1].plot(kind="barh", color="#55A868")
    plt.title("Top 10 Products by Quantity Sold")
    plt.xlabel("Total quantity")
    savefig("top_products.png")

    # Revenue trend over time (monthly)
    monthly = (
        df.set_index("InvoiceDate")["TotalPrice"].resample("ME").sum()
    )
    plt.figure(figsize=(10, 5))
    monthly.plot(marker="o", color="#C44E52")
    plt.title("Monthly Revenue Over Time")
    plt.ylabel("Revenue")
    plt.xlabel("Month")
    savefig("revenue_trend.png")

    # Monetary distribution per transaction (log scale)
    invoice_value = df.groupby("InvoiceNo")["TotalPrice"].sum()
    plt.figure(figsize=(9, 5))
    plt.hist(invoice_value.clip(upper=invoice_value.quantile(0.99)), bins=50,
             color="#8172B3")
    plt.title("Monetary Distribution per Transaction (99th pct capped)")
    plt.xlabel("Invoice value")
    plt.ylabel("Count")
    savefig("transaction_value_dist.png")

    # Monetary distribution per customer (log scale)
    customer_value = df.groupby("CustomerID")["TotalPrice"].sum()
    plt.figure(figsize=(9, 5))
    plt.hist(customer_value.clip(upper=customer_value.quantile(0.99)), bins=50,
             color="#CCB974")
    plt.title("Monetary Distribution per Customer (99th pct capped)")
    plt.xlabel("Total spend per customer")
    plt.ylabel("Count")
    savefig("customer_value_dist.png")

    # ---------------- RFM ----------------
    print("3/6  Building RFM table ...")
    rfm = build_rfm(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                              ["#4C72B0", "#55A868", "#C44E52"]):
        data = rfm[col].clip(upper=rfm[col].quantile(0.99))
        ax.hist(data, bins=40, color=color)
        ax.set_title(f"{col} distribution")
    fig.suptitle("RFM Distributions (99th pct capped)")
    savefig("rfm_distributions.png")

    # ---------------- choose k ----------------
    print("4/6  Evaluating cluster counts ...")
    scaled, scaler = scale_rfm(rfm)
    metrics = evaluate_k(scaled, range(2, 11))
    metrics.to_csv(os.path.join("reports", "cluster_metrics.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(metrics["k"], metrics["inertia"], "o-", color="#4C72B0", label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(metrics["k"], metrics["silhouette"], "s--", color="#C44E52",
             label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#C44E52")
    plt.title("Elbow & Silhouette for k selection")
    savefig("elbow_silhouette.png")

    # The brief defines four business segments (High-Value, Regular,
    # Occasional, At-Risk), so we use k=4. We still report silhouette for
    # the full range above for transparency.
    best_k = 4
    sil_at_k = float(metrics.loc[metrics["k"] == best_k, "silhouette"].iloc[0])
    print(f"     using k = {best_k} (silhouette = {sil_at_k:.3f})")

    # ---------------- cluster + label ----------------
    print("5/6  Fitting final KMeans and labeling segments ...")
    km = fit_kmeans(scaled, best_k)
    rfm["Cluster"] = km.labels_
    label_map = label_segments(rfm, "Cluster")
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    profile["Customers"] = rfm["Segment"].value_counts()
    profile.to_csv(os.path.join("reports", "segment_profiles.csv"))
    print(profile)

    # Cluster scatter: Recency vs Monetary, sized by frequency
    plt.figure(figsize=(9, 6))
    for seg in rfm["Segment"].unique():
        sub = rfm[rfm["Segment"] == seg]
        plt.scatter(sub["Recency"], sub["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)),
                    s=10, alpha=0.5, label=seg)
    plt.xlabel("Recency (days)")
    plt.ylabel("Monetary")
    plt.title("Customer Segments: Recency vs Monetary")
    plt.legend()
    savefig("cluster_scatter.png")

    rfm.to_csv(os.path.join("reports", "rfm_segments.csv"), index=False)

    # ---------------- recommender ----------------
    print("6/6  Building item-based similarity matrix ...")
    sim_df, code2name, name2code = build_similarity(df, min_customers=5)
    print(f"     similarity matrix: {sim_df.shape}")
    neighbors = build_top_neighbors(sim_df, code2name, top_n=20)

    # Product similarity heatmap for the top-N most popular products
    top_codes = (
        df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
        .index
    )
    top_codes = [c for c in top_codes if c in sim_df.index][:20]
    heat = sim_df.loc[top_codes, top_codes]
    labels = [str(code2name.get(c, c))[:22] for c in top_codes]
    plt.figure(figsize=(11, 9))
    if sns_ok:
        sns.heatmap(heat, xticklabels=labels, yticklabels=labels, cmap="viridis",
                    square=True, cbar_kws={"label": "cosine similarity"})
    else:
        plt.imshow(heat.values, cmap="viridis", aspect="auto")
        plt.colorbar(label="cosine similarity")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
    plt.title("Product Similarity Heatmap (top 20 products)")
    savefig("similarity_heatmap.png")

    # ---------------- save artifacts ----------------
    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_map, os.path.join(MODELS_DIR, "segment_labels.pkl"))
    joblib.dump(
        {"neighbors": neighbors, "code2name": code2name, "name2code": name2code},
        os.path.join(MODELS_DIR, "recommender.pkl"),
    )

    print("\nDone. Artifacts saved to models/ and figures to reports/figures/.")


if __name__ == "__main__":
    main()
