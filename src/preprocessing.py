"""
Data loading and cleaning for the Shopper Spectrum project.

Cleaning rules (from the project brief):
    - Drop rows with a missing CustomerID
    - Exclude cancelled invoices (InvoiceNo starting with 'C')
    - Remove rows with quantity <= 0 or unit price <= 0
    - Drop exact duplicate rows
"""

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw online retail CSV and parse the invoice date."""
    df = pd.read_csv(path, dtype={"InvoiceNo": "str", "StockCode": "str"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules and add a TotalPrice column."""
    df = df.copy()

    # 1. Drop missing customer IDs
    df = df[df["CustomerID"].notna()]

    # 2. Exclude cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 3. Remove non-positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 4. Drop exact duplicates and rows with no valid date
    df = df[df["InvoiceDate"].notna()]
    df = df.drop_duplicates()

    # Tidy types and add line-level revenue
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].astype(str).str.strip()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)


def load_and_clean(path: str) -> pd.DataFrame:
    """Convenience wrapper: load the raw CSV and return a cleaned frame."""
    return clean(load_raw(path))


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail.csv"
    raw = load_raw(src)
    cleaned = load_and_clean(src)
    print(f"Raw rows:     {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Customers:    {cleaned['CustomerID'].nunique():,}")
    print(f"Products:     {cleaned['StockCode'].nunique():,}")
