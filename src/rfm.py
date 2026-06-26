"""
RFM feature engineering.

Recency  = (reference date - customer's last purchase date) in days
Frequency = number of distinct invoices per customer
Monetary  = total amount spent by the customer
"""

import pandas as pd


def build_rfm(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Build a per-customer RFM table from cleaned transaction data.

    Parameters
    ----------
    df : cleaned transactions with InvoiceDate, InvoiceNo, CustomerID, TotalPrice
    reference_date : the "today" used to compute recency. Defaults to one day
        after the latest invoice in the data (standard RFM convention).
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    # Guard against any zero/negative monetary values that survive cleaning
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm


if __name__ == "__main__":
    from preprocessing import load_and_clean

    df = load_and_clean("data/online_retail.csv")
    rfm = build_rfm(df)
    print(rfm.head())
    print(rfm.describe())
