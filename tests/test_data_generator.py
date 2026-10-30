"""Unit tests for foodservice data generator."""

import pandas as pd

from freshcast.data.generator import FoodserviceDataGenerator


def test_generator_output_schema_and_types():
    generator = FoodserviceDataGenerator()
    df = generator.generate()

    assert not df.empty
    assert "date" in df.columns
    assert "sales_cases" in df.columns
    assert "dc_id" in df.columns
    assert "sku_id" in df.columns
    assert "shelf_life_days" in df.columns

    # Verify dates are valid and sorted
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    # Sales counts should be non-negative integers
    assert (df["sales_cases"] >= 0).all()
    assert (df["sales_cases"] % 1 == 0).all()


def test_generator_dc_and_sku_coverage():
    generator = FoodserviceDataGenerator()
    df = generator.generate()

    unique_dcs = df["dc_id"].unique()
    unique_skus = df["sku_id"].unique()

    assert len(unique_dcs) == 3
    assert len(unique_skus) == 10
    assert "DC_BOS" in unique_dcs
    assert "SKU_BEEF_SIRLOIN" in unique_skus
