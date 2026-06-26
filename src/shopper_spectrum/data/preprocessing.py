"""Transaction cleaning, driven by the preprocessing config block."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("shopper_spectrum.data.preprocessing")


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Clean raw transactions according to the config.

    Steps (each toggleable via config):
        - drop rows with missing CustomerID
        - exclude cancelled invoices (InvoiceNo starting with the cancel prefix)
        - keep Quantity >= min_quantity and UnitPrice >= min_unit_price
        - drop rows with an unparseable date
        - drop exact duplicates
    Adds a TotalPrice column (Quantity * UnitPrice).
    """
    n0 = len(df)
    df = df.copy()

    if cfg.get("drop_missing_customer", True):
        df = df[df["CustomerID"].notna()]

    if cfg.get("exclude_cancelled", True):
        prefix = cfg.get("cancel_prefix", "C")
        df = df[~df["InvoiceNo"].astype(str).str.startswith(prefix)]

    df = df[df["Quantity"] >= cfg.get("min_quantity", 1)]
    df = df[df["UnitPrice"] >= cfg.get("min_unit_price", 0.01)]
    df = df[df["InvoiceDate"].notna()]

    if cfg.get("drop_duplicates", True):
        df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].astype(str).str.strip()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df = df.reset_index(drop=True)
    logger.info("Cleaned data: %d -> %d rows (%.1f%% kept)",
                n0, len(df), 100 * len(df) / n0)
    return df
