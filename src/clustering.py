"""
KMeans clustering on standardized RFM features, plus a rule for turning
cluster centroids into human-readable segment labels.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def rfm_to_features(rfm: pd.DataFrame) -> np.ndarray:
    """
    Turn raw R, F, M into model features.

    Frequency and Monetary are heavily right-skewed, which makes raw
    StandardScaler isolate a handful of outliers into their own cluster.
    A log1p transform compresses that skew so KMeans finds balanced,
    business-meaningful segments. Recency is left as-is (already moderate).
    """
    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    feats["Frequency"] = np.log1p(feats["Frequency"])
    feats["Monetary"] = np.log1p(feats["Monetary"])
    return feats.values


def scale_rfm(rfm: pd.DataFrame):
    """Log-transform skewed features then standardize. Returns (scaled, scaler)."""
    feats = rfm_to_features(rfm)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    return scaled, scaler


def evaluate_k(scaled, k_range=range(2, 11)):
    """Compute inertia (elbow) and silhouette score for a range of k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(scaled, n_clusters: int):
    """Fit a KMeans model and return it."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled)
    return km


def label_segments(rfm: pd.DataFrame, cluster_col="Cluster") -> dict:
    """
    Map each cluster id to a segment label by ranking the cluster's mean RFM.

    Scoring idea: a good customer has LOW recency, HIGH frequency, HIGH
    monetary. We rank clusters on each axis and combine into a single score,
    then assign the four standard labels by that score.
    """
    profile = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

    # Higher rank = better. Recency is reversed (lower is better).
    r_rank = profile["Recency"].rank(ascending=False)      # low recency -> high rank
    f_rank = profile["Frequency"].rank(ascending=True)
    m_rank = profile["Monetary"].rank(ascending=True)
    score = r_rank + f_rank + m_rank

    ordered = score.sort_values(ascending=False).index.tolist()

    # Number of labels adapts to the number of clusters found.
    label_pool = ["High-Value", "Regular", "Occasional", "At-Risk"]
    n = len(ordered)
    if n <= len(label_pool):
        chosen = label_pool[:n]
    else:
        chosen = label_pool + [f"Segment-{i}" for i in range(len(label_pool), n)]

    return {cluster_id: chosen[i] for i, cluster_id in enumerate(ordered)}


if __name__ == "__main__":
    from preprocessing import load_and_clean
    from rfm import build_rfm

    df = load_and_clean("data/online_retail.csv")
    rfm = build_rfm(df)
    scaled, _ = scale_rfm(rfm)
    print(evaluate_k(scaled))
