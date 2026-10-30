"""Unit tests for forecasting models."""

import numpy as np
import pandas as pd

from freshcast.models.baselines import MovingAverageForecaster, NaiveSeasonalForecaster
from freshcast.models.gbdt import LightGBMForecaster, XGBoostForecaster


def test_naive_seasonal_forecaster():
    X = pd.DataFrame({"lag_7": [10.0, 20.0, 30.0]})
    y = pd.Series([12.0, 22.0, 32.0])
    model = NaiveSeasonalForecaster(lag=7)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, np.array([10.0, 20.0, 30.0]))


def test_moving_average_forecaster():
    X = pd.DataFrame({"rolling_mean_7": [15.0, 25.0, 35.0]})
    y = pd.Series([10.0, 20.0, 30.0])
    model = MovingAverageForecaster(window=7)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, np.array([15.0, 25.0, 35.0]))


def test_lightgbm_and_xgboost_fit_predict():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "lag_1": np.random.uniform(50, 100, size=n),
            "lag_7": np.random.uniform(50, 100, size=n),
            "day_of_week": np.random.randint(0, 7, size=n),
        }
    )
    y = pd.Series(X["lag_1"] * 0.7 + X["lag_7"] * 0.3 + np.random.normal(0, 2, size=n))

    # Test LightGBM
    lgb_model = LightGBMForecaster(params={"n_estimators": 20})
    lgb_model.fit(X, y)
    lgb_preds = lgb_model.predict(X)
    assert len(lgb_preds) == n
    assert (lgb_preds >= 0).all()

    # Test XGBoost
    xgb_model = XGBoostForecaster(params={"n_estimators": 20})
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict(X)
    assert len(xgb_preds) == n
    assert (xgb_preds >= 0).all()
