"""Statistical time-series forecasting with Facebook Prophet."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from freshcast.models.base import BaseDemandForecaster

# Suppress cmdstanpy / prophet verbose output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ProphetForecaster(BaseDemandForecaster):
    """Multi-series wrapper around Facebook Prophet.

    Fits independent Prophet trend & seasonal components per (DC, SKU) series.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(name="Facebook Prophet", params=params)
        self.models: dict[str, Any] = {}
        self.global_mean: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_series: pd.Series | None = None,
        series_id_series: pd.Series | None = None,
        **kwargs,
    ) -> ProphetForecaster:
        if Prophet is None:
            raise ImportError("Prophet is not installed.")

        self.global_mean = float(y.mean()) if len(y) > 0 else 0.0

        if date_series is None and "date" in X.columns:
            date_series = X["date"]

        # If series_id not provided, look for 'series_id' or combine dc_id and sku_id
        if series_id_series is None:
            if "series_id" in X.columns:
                series_id_series = X["series_id"]
            elif "dc_id" in X.columns and "sku_id" in X.columns:
                series_id_series = X["dc_id"].astype(str) + "_" + X["sku_id"].astype(str)
            else:
                series_id_series = pd.Series(["global"] * len(X), index=X.index)

        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(date_series),
                "y": y.values,
                "series_id": series_id_series.values,
            }
        )

        # Fit a Prophet model per unique series
        for sid, group in df.groupby("series_id"):
            m = Prophet(
                growth="linear",
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=self.params.get("seasonality_mode", "multiplicative"),
                changepoint_prior_scale=self.params.get("changepoint_prior_scale", 0.05),
            )
            # Add US holidays
            m.add_country_holidays(country_name="US")
            m.fit(group[["ds", "y"]])
            self.models[sid] = m

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        date_series: pd.Series | None = None,
        series_id_series: pd.Series | None = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ProphetForecaster must be fitted before calling predict.")

        if date_series is None and "date" in X.columns:
            date_series = X["date"]

        if series_id_series is None:
            if "series_id" in X.columns:
                series_id_series = X["series_id"]
            elif "dc_id" in X.columns and "sku_id" in X.columns:
                series_id_series = X["dc_id"].astype(str) + "_" + X["sku_id"].astype(str)
            else:
                series_id_series = pd.Series(["global"] * len(X), index=X.index)

        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(date_series),
                "series_id": series_id_series.values,
            },
            index=X.index,
        )

        preds = np.zeros(len(X), dtype=float)

        for sid, group in df.groupby("series_id"):
            idx = group.index
            if sid in self.models:
                m = self.models[sid]
                forecast = m.predict(group[["ds"]])
                preds[idx] = np.clip(forecast["yhat"].values, 0, None)
            else:
                preds[idx] = self.global_mean

        return preds
