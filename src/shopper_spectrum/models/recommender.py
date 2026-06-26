"""
Item-based collaborative filtering recommender.

Builds a product-product cosine similarity matrix from a customer-product
quantity matrix, compresses it to the top-N neighbours per product, and
provides a leave-one-out hit-rate evaluation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("shopper_spectrum.models.recommender")


def build_customer_product_matrix(df: pd.DataFrame, min_customers: int = 5):
    """
    Build a customer x product quantity matrix, keeping only products bought by
    at least `min_customers` distinct customers. Returns (matrix, code2name).
    """
    code2name = (
        df.groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    pop = df.groupby("StockCode")["CustomerID"].nunique()
    keep = pop[pop >= min_customers].index
    sub = df[df["StockCode"].isin(keep)]
    matrix = sub.pivot_table(
        index="CustomerID", columns="StockCode",
        values="Quantity", aggfunc="sum", fill_value=0,
    )
    logger.info("Customer-product matrix: %s", matrix.shape)
    return matrix, code2name


def build_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between products (matrix columns)."""
    sim = cosine_similarity(csr_matrix(matrix.values).T)
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def build_top_neighbors(sim_df: pd.DataFrame, top_n: int = 20) -> dict:
    """Compress the dense matrix into a top-N neighbour dict per product."""
    neighbors = {}
    arr = sim_df.values
    codes = sim_df.index.to_numpy()
    for i, code in enumerate(codes):
        row = arr[i].copy()
        row[i] = -1.0
        idx = np.argpartition(row, -top_n)[-top_n:]
        idx = idx[np.argsort(row[idx])[::-1]]
        neighbors[code] = [(codes[j], round(float(row[j]), 4)) for j in idx]
    return neighbors


def build_name_index(code2name: dict, codes) -> dict:
    """Map lowercased product names to stock codes for lookup."""
    return {str(code2name.get(c, c)).strip().lower(): c for c in codes}


def recommend(product_name, neighbors, code2name, name2code, n: int = 5):
    """Return the top-N similar products to a product name (substring fallback)."""
    query = str(product_name).strip().lower()
    code = name2code.get(query)
    if code is None:
        matches = [name for name in name2code if query in name]
        if not matches:
            return None  # no match at all
        code = name2code[matches[0]]
    if code not in neighbors:
        return []
    return [
        {"stock_code": c, "product": code2name.get(c, c), "score": s}
        for c, s in neighbors[code][:n]
    ]


def evaluate_hit_rate(matrix: pd.DataFrame, neighbors: dict,
                      sample: int = 500, n: int = 5, seed: int = 42) -> float:
    """
    Leave-one-out hit rate: for sampled customers with >=2 purchased products,
    hide one product and check whether it appears in the recommendations
    generated from another product they bought.
    """
    rng = np.random.default_rng(seed)
    bought = {cust: list(row[row > 0].index)
              for cust, row in matrix.iterrows()}
    eligible = [c for c, items in bought.items() if len(items) >= 2]
    if not eligible:
        return float("nan")
    picked = rng.choice(eligible, size=min(sample, len(eligible)), replace=False)

    hits = 0
    for cust in picked:
        items = bought[cust]
        held_out = items[rng.integers(len(items))]
        seed_item = next(it for it in items if it != held_out)
        recs = {c for c, _ in neighbors.get(seed_item, [])[:n]}
        if held_out in recs:
            hits += 1
    rate = hits / len(picked)
    logger.info("Recommender hit-rate@%d: %.3f (n=%d)", n, rate, len(picked))
    return rate
