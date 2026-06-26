"""Data loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("shopper_spectrum.data.loader")


def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Load the raw online-retail CSV and parse the invoice date.

    Parameters
    ----------
    path : path to the raw CSV.

    Returns
    -------
    DataFrame with InvoiceDate parsed to datetime.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Place online_retail.csv in data/raw/."
        )
    df = pd.read_csv(path, dtype={"InvoiceNo": "str", "StockCode": "str"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    logger.info("Loaded raw data: %d rows", len(df))
    return df
