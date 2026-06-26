"""Plotting helpers. All figures are saved to the configured figures dir."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("shopper_spectrum.visualization.plots")

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    _SNS = True
except Exception:  # pragma: no cover
    _SNS = False

PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]


def _save(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / name, dpi=120, bbox_inches="tight")
    plt.close()


def eda_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    """Generate the core EDA figures required by the brief."""
    fig_dir = Path(fig_dir)

    df["Country"].value_counts().head(10).iloc[::-1].plot(
        kind="barh", color=PALETTE[0], figsize=(9, 5))
    plt.title("Top 10 Countries by Transaction Volume"); plt.xlabel("Line items")
    _save(fig_dir, "top_countries.png")

    (df.groupby("Description")["Quantity"].sum().sort_values(ascending=False)
       .head(10).iloc[::-1].plot(kind="barh", color=PALETTE[1], figsize=(9, 5)))
    plt.title("Top 10 Products by Quantity Sold"); plt.xlabel("Total quantity")
    _save(fig_dir, "top_products.png")

    monthly = df.set_index("InvoiceDate")["TotalPrice"].resample("ME").sum()
    monthly.plot(marker="o", color=PALETTE[2], figsize=(10, 5))
    plt.title("Monthly Revenue Over Time"); plt.ylabel("Revenue")
    _save(fig_dir, "revenue_trend.png")

    inv = df.groupby("InvoiceNo")["TotalPrice"].sum()
    plt.figure(figsize=(9, 5))
    plt.hist(inv.clip(upper=inv.quantile(0.99)), bins=50, color=PALETTE[3])
    plt.title("Monetary Distribution per Transaction"); plt.xlabel("Invoice value")
    _save(fig_dir, "transaction_value_dist.png")

    cust = df.groupby("CustomerID")["TotalPrice"].sum()
    plt.figure(figsize=(9, 5))
    plt.hist(cust.clip(upper=cust.quantile(0.99)), bins=50, color=PALETTE[4])
    plt.title("Monetary Distribution per Customer"); plt.xlabel("Total spend")
    _save(fig_dir, "customer_value_dist.png")


def rfm_distributions(rfm: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"], PALETTE):
        ax.hist(rfm[col].clip(upper=rfm[col].quantile(0.99)), bins=40, color=color)
        ax.set_title(f"{col} distribution")
    fig.suptitle("RFM Distributions (99th pct capped)")
    _save(Path(fig_dir), "rfm_distributions.png")


def elbow_and_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Plot elbow + silhouette for the chosen (kmeans) algorithm."""
    km = metrics[metrics["algorithm"] == "kmeans"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(km["k"], km["inertia"], "o-", color=PALETTE[0], label="Inertia")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia", color=PALETTE[0])
    ax2 = ax1.twinx()
    ax2.plot(km["k"], km["silhouette"], "s--", color=PALETTE[2], label="Silhouette")
    ax2.set_ylabel("Silhouette", color=PALETTE[2])
    plt.title("Elbow & Silhouette (KMeans)")
    _save(Path(fig_dir), "elbow_silhouette.png")


def algorithm_comparison(metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Silhouette score per algorithm across k."""
    plt.figure(figsize=(9, 5))
    for i, algo in enumerate(metrics["algorithm"].unique()):
        sub = metrics[metrics["algorithm"] == algo]
        plt.plot(sub["k"], sub["silhouette"], "o-", color=PALETTE[i % len(PALETTE)],
                 label=algo)
    plt.xlabel("k"); plt.ylabel("Silhouette score")
    plt.title("Algorithm Comparison (Silhouette vs k)"); plt.legend()
    _save(Path(fig_dir), "algorithm_comparison.png")


def cluster_scatter(rfm: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for i, seg in enumerate(rfm["Segment"].unique()):
        sub = rfm[rfm["Segment"] == seg]
        plt.scatter(sub["Recency"],
                    sub["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)),
                    s=10, alpha=0.5, color=PALETTE[i % len(PALETTE)], label=seg)
    plt.xlabel("Recency (days)"); plt.ylabel("Monetary")
    plt.title("Customer Segments: Recency vs Monetary"); plt.legend()
    _save(Path(fig_dir), "cluster_scatter.png")


def cluster_scatter_3d(rfm: pd.DataFrame, fig_dir: Path) -> None:
    """3D RFM scatter coloured by segment."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, seg in enumerate(rfm["Segment"].unique()):
        sub = rfm[rfm["Segment"] == seg]
        ax.scatter(sub["Recency"], sub["Frequency"].clip(upper=rfm["Frequency"].quantile(0.99)),
                   sub["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)),
                   s=8, alpha=0.5, color=PALETTE[i % len(PALETTE)], label=seg)
    ax.set_xlabel("Recency"); ax.set_ylabel("Frequency"); ax.set_zlabel("Monetary")
    ax.set_title("3D RFM Customer Segments"); ax.legend()
    _save(Path(fig_dir), "cluster_scatter_3d.png")


def similarity_heatmap(sim_df: pd.DataFrame, code2name: dict, df: pd.DataFrame,
                       fig_dir: Path, top: int = 20) -> None:
    codes = (df.groupby("StockCode")["Quantity"].sum()
               .sort_values(ascending=False).index)
    codes = [c for c in codes if c in sim_df.index][:top]
    heat = sim_df.loc[codes, codes]
    labels = [str(code2name.get(c, c))[:22] for c in codes]
    plt.figure(figsize=(11, 9))
    if _SNS:
        sns.heatmap(heat, xticklabels=labels, yticklabels=labels, cmap="viridis",
                    square=True, cbar_kws={"label": "cosine similarity"})
    else:
        plt.imshow(heat.values, cmap="viridis", aspect="auto")
        plt.colorbar(label="cosine similarity")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
    plt.title("Product Similarity Heatmap (top 20 products)")
    _save(Path(fig_dir), "similarity_heatmap.png")
