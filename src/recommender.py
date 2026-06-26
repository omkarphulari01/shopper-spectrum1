"""
Item-based collaborative filtering.

We build a Customer x Product matrix (quantity purchased), then compute
cosine similarity between products. Given a product name, we return the
top-N most similar products.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(df: pd.DataFrame, min_customers: int = 5):
    """
    Build the item-item cosine similarity matrix.

    To keep the matrix manageable and the similarities meaningful, we keep
    only products purchased by at least `min_customers` distinct customers.

    Returns
    -------
    sim_df    : DataFrame of shape (n_products, n_products) with StockCode index
    code2name : dict StockCode -> most common Description
    name2code : dict lowercase Description -> StockCode
    """
    # Most frequent description per stock code (descriptions can vary slightly)
    code2name = (
        df.groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    # Filter to popular-enough products
    pop = df.groupby("StockCode")["CustomerID"].nunique()
    keep = pop[pop >= min_customers].index
    sub = df[df["StockCode"].isin(keep)]

    # Customer x Product matrix of total quantity
    matrix = sub.pivot_table(
        index="CustomerID",
        columns="StockCode",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    # Cosine similarity between products (columns)
    sparse = csr_matrix(matrix.values)
    sim = cosine_similarity(sparse.T)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

    name2code = {}
    for code in matrix.columns:
        name = code2name.get(code, code)
        name2code[str(name).strip().lower()] = code

    return sim_df, code2name, name2code


def build_top_neighbors(sim_df, code2name, top_n: int = 20):
    """
    Compress the full similarity matrix into a compact dict of the top-N
    neighbors per product. This is all the app needs and shrinks the saved
    artifact from hundreds of MB to a few MB.
    """
    neighbors = {}
    arr = sim_df.values
    codes = sim_df.index.to_numpy()
    for i, code in enumerate(codes):
        row = arr[i].copy()
        row[i] = -1.0  # exclude self
        idx = np.argpartition(row, -top_n)[-top_n:]
        idx = idx[np.argsort(row[idx])[::-1]]
        neighbors[code] = [(codes[j], round(float(row[j]), 4)) for j in idx]
    return neighbors


def recommend_compact(product_name, neighbors, code2name, name2code, n=5):
    """Top-N recommendation using the compact neighbor dict."""
    query = str(product_name).strip().lower()
    code = name2code.get(query)
    if code is None:
        matches = [name for name in name2code if query in name]
        if not matches:
            return []
        code = name2code[matches[0]]
    if code not in neighbors:
        return []
    return [
        {"stock_code": c, "product": code2name.get(c, c), "score": s}
        for c, s in neighbors[code][:n]
    ]


def recommend(product_name, sim_df, code2name, name2code, n=5):
    """
    Return the top-N similar products to a given product name.

    Matching is case-insensitive with a substring fallback, so users do not
    need to type the description exactly.
    """
    query = str(product_name).strip().lower()

    code = name2code.get(query)
    if code is None:
        # substring fallback
        matches = [name for name in name2code if query in name]
        if not matches:
            return []
        code = name2code[matches[0]]

    if code not in sim_df.index:
        return []

    scores = sim_df[code].drop(labels=[code]).sort_values(ascending=False).head(n)
    return [
        {"stock_code": c, "product": code2name.get(c, c), "score": round(float(s), 3)}
        for c, s in scores.items()
    ]


if __name__ == "__main__":
    from preprocessing import load_and_clean

    df = load_and_clean("data/online_retail.csv")
    sim_df, code2name, name2code = build_similarity(df)
    print("Matrix:", sim_df.shape)
    example = code2name[sim_df.index[0]]
    print("Recommendations for:", example)
    for r in recommend(example, sim_df, code2name, name2code):
        print("  ", r)
