"""Unit tests for feature engineering and leakage prevention."""

import numpy as np
import pandas as pd

from freshcast.features.calendar import CalendarFeatureExtractor
from freshcast.features.lags import LagFeatureExtractor
from freshcast.features.pipeline import TimeSeriesFeaturePipeline


def test_calendar_feature_extractor():
    dates = pd.date_range("2023-01-01", periods=14, freq="D")
    df = pd.DataFrame({"date": dates})
    extractor = CalendarFeatureExtractor(country="US")
    res = extractor.transform(df)

    assert "day_of_week" in res.columns
    assert "sin_dow" in res.columns
    assert "cos_dow" in res.columns
    assert "is_monday_restock" in res.columns
    assert "is_weekend_prep" in res.columns
    assert "is_holiday" in res.columns
    assert res["sin_dow"].between(-1.0, 1.0).all()


def test_lag_feature_extractor_no_lookahead_leakage():
    # Construct synthetic series: sales = 10, 20, 30, 40, 50
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "dc_id": "DC_1",
            "sku_id": "SKU_1",
            "sales_cases": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    extractor = LagFeatureExtractor(lags=[1, 2], rolling_windows=[2])
    res = extractor.transform(df)

    # At row index 2 (date=2023-01-03, current sales=30):
    # lag_1 MUST be 20.0
    # lag_2 MUST be 10.0
    # rolling_mean_2 on shifted series MUST be mean([10, 20]) = 15.0
    row_2 = res.iloc[2]
    assert row_2["sales_cases"] == 30.0
    assert row_2["lag_1"] == 20.0
    assert row_2["lag_2"] == 10.0
    assert np.isclose(row_2["rolling_mean_2"], 15.0)


def test_pipeline_transform_end_to_end():
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "dc_id": "DC_BOS",
            "sku_id": "SKU_BEEF",
            "category": "Meat",
            "base_unit_price": 100.0,
            "actual_unit_price": 90.0,
            "discount_pct": 0.10,
            "is_promo": 1,
            "sales_cases": np.random.randint(50, 150, size=40),
        }
    )

    pipeline = TimeSeriesFeaturePipeline(lags=[1, 7], rolling_windows=[7])
    X, y, feature_cols = pipeline.prepare_train_matrices(df)

    assert len(X) == len(y)
    assert not X.isnull().values.any()
    assert "lag_7" in feature_cols
    assert "rolling_mean_7" in feature_cols
