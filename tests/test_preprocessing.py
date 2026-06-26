"""Tests for data cleaning rules."""

from shopper_spectrum.data.preprocessing import clean


def test_drops_missing_customer(raw_df, prep_cfg):
    out = clean(raw_df, prep_cfg)
    assert out["CustomerID"].notna().all()
    # The Cherry row (customer None) must be gone
    assert "Cherry" not in out["Description"].values


def test_excludes_cancelled(raw_df, prep_cfg):
    out = clean(raw_df, prep_cfg)
    assert not out["InvoiceNo"].astype(str).str.startswith("C").any()


def test_removes_non_positive_quantity_and_price(raw_df, prep_cfg):
    out = clean(raw_df, prep_cfg)
    assert (out["Quantity"] >= 1).all()
    assert (out["UnitPrice"] >= 0.01).all()


def test_adds_total_price(raw_df, prep_cfg):
    out = clean(raw_df, prep_cfg)
    assert "TotalPrice" in out.columns
    assert (out["TotalPrice"] == out["Quantity"] * out["UnitPrice"]).all()


def test_customer_id_is_int(raw_df, prep_cfg):
    out = clean(raw_df, prep_cfg)
    assert out["CustomerID"].dtype.kind == "i"
