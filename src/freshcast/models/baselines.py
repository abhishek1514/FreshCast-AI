"""Baseline benchmark models for time-series demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from freshcast.models.base import BaseDemandForecaster


class NaiveSeasonalForecaster(BaseDemandForecaster):
    """Predicts current demand using seasonal lag t-7 (same day of last week)."""

    def __init__(self, lag: int = 7):
        super().__init__(name=f"Naive Seasonal (t-{lag})")
        self.lag = lag

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> NaiveSeasonalForecaster:
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = f"lag_{self.lag}"
        if lag_col in X.columns:
            return np.nan_to_num(X[lag_col].to_numpy(), nan=0.0)
        # Fallback to lag_1 or mean if specific lag is missing
        if "lag_1" in X.columns:
            return np.nan_to_num(X["lag_1"].to_numpy(), nan=0.0)
        return np.zeros(len(X))


class MovingAverageForecaster(BaseDemandForecaster):
    """Predicts current demand using recent rolling mean (e.g. 7-day or 14-day average)."""

    def __init__(self, window: int = 7):
        super().__init__(name=f"Moving Average ({window}d)")
        self.window = window
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> MovingAverageForecaster:
        self.global_mean = float(y.mean())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rolling_col = f"rolling_mean_{self.window}"
        if rolling_col in X.columns:
            preds = X[rolling_col].to_numpy()
            return np.where(np.isnan(preds), self.global_mean, preds)
        return np.full(len(X), self.global_mean)
