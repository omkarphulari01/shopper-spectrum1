"""RFM feature engineering, including classic 1-5 quartile scoring."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("shopper_spectrum.features.rfm")


def build_rfm(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Build a per-customer RFM table.

    Recency   = days between the reference date and the customer's last purchase
    Frequency = number of distinct invoices
    Monetary  = total spend

    Also adds R/F/M quartile scores (1-5) and a combined RFM_Score, which is a
    classic, interpretable complement to the unsupervised clustering.
    """
    cfg = cfg or {}
    ref = cfg.get("reference_date")
    reference_date = (
        pd.to_datetime(ref) if ref else df["InvoiceDate"].max() + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)

    _add_quartile_scores(rfm)
    logger.info("Built RFM table for %d customers", len(rfm))
    return rfm


def _add_quartile_scores(rfm: pd.DataFrame) -> None:
    """Add R, F, M scores (1-5) and a combined RFM_Score in place."""
    # Recency: lower is better -> reverse the labels
    rfm["R_Score"] = _safe_qcut(rfm["Recency"], reverse=True)
    rfm["F_Score"] = _safe_qcut(rfm["Frequency"].rank(method="first"), reverse=False)
    rfm["M_Score"] = _safe_qcut(rfm["Monetary"], reverse=False)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(int)
        + rfm["F_Score"].astype(int)
        + rfm["M_Score"].astype(int)
    )


def _safe_qcut(series: pd.Series, reverse: bool, q: int = 5) -> pd.Series:
    """Quantile-bin into 1..q, robust to ties/duplicate edges."""
    labels = list(range(1, q + 1))
    if reverse:
        labels = labels[::-1]
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(int)
    except ValueError:
        # Too few distinct values for q bins — rank-based fallback
        ranked = series.rank(method="first")
        binned = np.ceil(ranked / len(ranked) * q).clip(1, q).astype(int)
        return (q + 1 - binned) if reverse else binned
