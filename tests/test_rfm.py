"""Tests for RFM feature engineering."""

from shopper_spectrum.data.preprocessing import clean
from shopper_spectrum.features.rfm import build_rfm


def test_rfm_columns(raw_df, prep_cfg):
    rfm = build_rfm(clean(raw_df, prep_cfg))
    for col in ["Recency", "Frequency", "Monetary", "RFM_Score"]:
        assert col in rfm.columns


def test_frequency_counts_distinct_invoices(raw_df, prep_cfg):
    rfm = build_rfm(clean(raw_df, prep_cfg))
    # Customer 10's invoice 6 has qty 0 (dropped in cleaning), leaving only
    # invoice 1 -> frequency 1.
    cust10 = rfm.loc[rfm["CustomerID"] == 10, "Frequency"].iloc[0]
    assert cust10 == 1


def test_monetary_is_positive(raw_df, prep_cfg):
    rfm = build_rfm(clean(raw_df, prep_cfg))
    assert (rfm["Monetary"] > 0).all()


def test_rfm_score_range(raw_df, prep_cfg):
    rfm = build_rfm(clean(raw_df, prep_cfg))
    assert rfm["R_Score"].between(1, 5).all()
    assert rfm["RFM_Score"].between(3, 15).all()
