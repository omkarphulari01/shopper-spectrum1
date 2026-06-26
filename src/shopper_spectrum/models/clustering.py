"""
Customer segmentation via clustering.

Supports comparing KMeans, Agglomerative (hierarchical) and Gaussian Mixture
models, selecting k by internal validation metrics, fitting a final model, and
turning cluster centroids into business-readable segment labels.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..evaluation.metrics import cluster_scores

logger = logging.getLogger("shopper_spectrum.models.clustering")

RFM_COLS = ["Recency", "Frequency", "Monetary"]


def build_features(rfm: pd.DataFrame, log_cols: list[str]) -> np.ndarray:
    """Apply log1p to skewed columns then return the raw R/F/M feature matrix."""
    feats = rfm[RFM_COLS].copy()
    for col in log_cols:
        if col in feats:
            feats[col] = np.log1p(feats[col])
    return feats.values


def scale_features(rfm: pd.DataFrame, log_cols: list[str]):
    """Log-transform + standardize. Returns (scaled_array, fitted_scaler)."""
    feats = build_features(rfm, log_cols)
    scaler = StandardScaler()
    return scaler.fit_transform(feats), scaler


def _fit_predict(algorithm: str, X: np.ndarray, k: int, random_state: int):
    """Fit one clustering algorithm and return its labels."""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        return model.fit_predict(X), model
    if algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k)
        return model.fit_predict(X), model
    if algorithm == "gmm":
        model = GaussianMixture(n_components=k, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model
    raise ValueError(f"Unknown algorithm: {algorithm}")


def compare_algorithms(
    X: np.ndarray, algorithms: list[str], k_range: range, random_state: int = 42
) -> pd.DataFrame:
    """
    Run every (algorithm, k) combination and record validation metrics plus
    KMeans inertia (for the elbow). Returns a tidy DataFrame.
    """
    rows = []
    for algo in algorithms:
        for k in k_range:
            labels, model = _fit_predict(algo, X, k, random_state)
            scores = cluster_scores(X, labels)
            inertia = getattr(model, "inertia_", np.nan)
            rows.append({"algorithm": algo, "k": k, "inertia": inertia, **scores})
    return pd.DataFrame(rows)


def fit_final(X: np.ndarray, algorithm: str, k: int, random_state: int = 42):
    """Fit the chosen final model and return (labels, model)."""
    logger.info("Fitting final model: %s (k=%d)", algorithm, k)
    return _fit_predict(algorithm, X, k, random_state)


def label_segments(rfm: pd.DataFrame, labels_pool: list[str],
                   cluster_col: str = "Cluster") -> dict[int, str]:
    """
    Map each cluster id to a segment label by ranking mean RFM.

    A strong customer has LOW recency, HIGH frequency, HIGH monetary. Each
    cluster gets a combined rank; clusters are then assigned labels from the
    pool in descending order of that rank.
    """
    profile = rfm.groupby(cluster_col)[RFM_COLS].mean()
    score = (
        profile["Recency"].rank(ascending=False)
        + profile["Frequency"].rank(ascending=True)
        + profile["Monetary"].rank(ascending=True)
    )
    ordered = score.sort_values(ascending=False).index.tolist()

    n = len(ordered)
    if n <= len(labels_pool):
        chosen = labels_pool[:n]
    else:
        chosen = labels_pool + [f"Segment-{i}" for i in range(len(labels_pool), n)]
    return {cid: chosen[i] for i, cid in enumerate(ordered)}
