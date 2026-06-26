"""Internal cluster-validation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def cluster_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Compute three complementary internal validation metrics.

    silhouette        : higher is better (-1..1)
    calinski_harabasz : higher is better
    davies_bouldin    : lower is better
    """
    n_labels = len(set(labels))
    if n_labels < 2:
        return {"silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }
