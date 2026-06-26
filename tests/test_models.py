"""Tests for clustering and the recommender."""

import numpy as np
import pandas as pd

from shopper_spectrum.data.preprocessing import clean
from shopper_spectrum.features.rfm import build_rfm
from shopper_spectrum.models import clustering as clu
from shopper_spectrum.models import recommender as rec


def test_scale_features_shape(raw_df, prep_cfg):
    rfm = build_rfm(clean(raw_df, prep_cfg))
    X, scaler = clu.scale_features(rfm, ["Frequency", "Monetary"])
    assert X.shape == (len(rfm), 3)
    # standardized columns ~ zero mean
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)


def test_label_segments_assigns_known_labels():
    rfm = pd.DataFrame({
        "Cluster": [0, 1, 2, 3],
        "Recency": [5, 50, 100, 300],
        "Frequency": [50, 10, 3, 1],
        "Monetary": [9000, 2000, 500, 100],
    })
    labels = clu.label_segments(rfm, ["High-Value", "Regular", "Occasional", "At-Risk"])
    # Cluster 0 (best RFM) should be High-Value
    assert labels[0] == "High-Value"
    # Cluster 3 (worst) should be At-Risk
    assert labels[3] == "At-Risk"


def test_recommender_returns_similar_products(raw_df, prep_cfg):
    df = clean(raw_df, prep_cfg)
    matrix, code2name = rec.build_customer_product_matrix(df, min_customers=1)
    sim_df = rec.build_similarity(matrix)
    neighbors = rec.build_top_neighbors(sim_df, top_n=2)
    name2code = rec.build_name_index(code2name, sim_df.index)
    out = rec.recommend("Apple", neighbors, code2name, name2code, n=2)
    assert out is not None
    assert isinstance(out, list)


def test_recommender_unknown_product_returns_none(raw_df, prep_cfg):
    df = clean(raw_df, prep_cfg)
    matrix, code2name = rec.build_customer_product_matrix(df, min_customers=1)
    sim_df = rec.build_similarity(matrix)
    neighbors = rec.build_top_neighbors(sim_df, top_n=2)
    name2code = rec.build_name_index(code2name, sim_df.index)
    assert rec.recommend("nonexistent-xyz", neighbors, code2name, name2code) is None
