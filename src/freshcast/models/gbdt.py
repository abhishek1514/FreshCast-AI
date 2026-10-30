"""Gradient Boosted Decision Tree (GBDT) Demand Forecasters: LightGBM and XGBoost."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from freshcast.models.base import BaseDemandForecaster

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseDemandForecaster):
    """LightGBM regressor optimized for multi-SKU retail and foodservice demand forecasting."""

    DEFAULT_PARAMS = {
        "objective": "regression_l1",  # L1 aligns directly to WAPE minimization
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "feature_fraction": 0.85,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        combined_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="LightGBM Regressor", params=combined_params)
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
        **kwargs,
    ) -> LightGBMForecaster:
        self.feature_names_ = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.params)

        fit_params: dict[str, Any] = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        # Ensure column alignment
        X_aligned = X[self.feature_names_]
        raw_preds = self.model.predict(X_aligned)
        return np.clip(raw_preds, 0, None)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        importance = self.model.feature_importances_
        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": importance,
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
        return df


class XGBoostForecaster(BaseDemandForecaster):
    """XGBoost regressor for time-series demand forecasting."""

    DEFAULT_PARAMS = {
        "objective": "reg:absoluteerror",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        combined_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="XGBoost Regressor", params=combined_params)
        self.model: xgb.XGBRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
        **kwargs,
    ) -> XGBoostForecaster:
        self.feature_names_ = list(X.columns)
        self.model = xgb.XGBRegressor(**self.params)

        fit_params: dict[str, Any] = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        X_aligned = X[self.feature_names_]
        raw_preds = self.model.predict(X_aligned)
        return np.clip(raw_preds, 0, None)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        importance = self.model.feature_importances_
        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": importance,
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
        return df
