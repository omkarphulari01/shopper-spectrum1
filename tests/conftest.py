"""Shared fixtures: a small synthetic transaction dataset."""

import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """A tiny raw-like frame exercising every cleaning rule."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "2", "C3", "4", "5", "6"],
            "StockCode": ["A", "B", "A", "A", "B", "C", "A"],
            "Description": ["Apple", "Banana", "Apple", "Apple", "Banana", "Cherry", "Apple"],
            "Quantity": [2, 1, 3, 5, -1, 4, 0],          # -1 and 0 dropped
            "InvoiceDate": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-02-01",
                 "2023-02-05", "2023-03-01", "2023-03-02", "2023-03-03"]
            ),
            "UnitPrice": [1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0],  # 0.0 price dropped
            "CustomerID": [10.0, 10.0, 11.0, 11.0, 12.0, None, 10.0],  # None dropped
            "Country": ["UK", "UK", "France", "France", "UK", "UK", "UK"],
        }
    )


@pytest.fixture
def prep_cfg():
    return {
        "drop_missing_customer": True,
        "exclude_cancelled": True,
        "cancel_prefix": "C",
        "min_quantity": 1,
        "min_unit_price": 0.01,
        "drop_duplicates": True,
    }
