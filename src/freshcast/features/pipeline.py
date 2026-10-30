"""Unified Time-Series Feature Pipeline."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from freshcast.features.calendar import CalendarFeatureExtractor
from freshcast.features.lags import LagFeatureExtractor

logger = logging.getLogger(__name__)


class TimeSeriesFeaturePipeline:
    """Orchestrates calendar, lag, promotional, and categorical encoding."""

    CATEGORICAL_COLS = ["dc_id", "sku_id", "category"]
    TARGET_COL = "sales_cases"

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        country: str = "US",
    ):
        self.calendar_extractor = CalendarFeatureExtractor(country=country)
        self.lag_extractor = LagFeatureExtractor(
            lags=lags,
            rolling_windows=rolling_windows,
            target_col=self.TARGET_COL,
            group_cols=["dc_id", "sku_id"],
        )
        self.label_encoders = {col: LabelEncoder() for col in self.CATEGORICAL_COLS}
        self.is_fitted = False
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> TimeSeriesFeaturePipeline:
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                self.label_encoders[col].fit(df[col].astype(str))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        if not self.is_fitted:
            self.fit(df)

        data = df.copy()

        # 1. Calendar Features
        data = self.calendar_extractor.transform(data, date_col="date")

        # 2. Lag and Rolling Features
        data = self.lag_extractor.transform(data)

        # 3. Price & Promotional Features
        if "actual_unit_price" in data.columns and "base_unit_price" in data.columns:
            data["relative_price"] = data["actual_unit_price"] / (data["base_unit_price"] + 1e-5)
        if "discount_pct" not in data.columns:
            data["discount_pct"] = 0.0
        if "is_promo" not in data.columns:
            data["is_promo"] = 0

        # 4. Categorical Encodings
        for col in self.CATEGORICAL_COLS:
            if col in data.columns:
                # Handle unseen categories smoothly
                classes = set(self.label_encoders[col].classes_)
                safe_series = (
                    data[col]
                    .astype(str)
                    .map(lambda x: (x if x in classes else self.label_encoders[col].classes_[0]))
                )
                data[f"{col}_encoded"] = self.label_encoders[col].transform(safe_series)

        # 5. Drop NA rows introduced by the maximum lag if requested
        if drop_na:
            max_lag = max(self.lag_extractor.lags)
            data = data.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)

        # Store engineered feature names (excluding metadata, identifiers and raw targets)
        exclude_cols = {
            "date",
            "dc_name",
            "sku_name",
            self.TARGET_COL,
            "dc_id",
            "sku_id",
            "category",
            "spoilage_cost_factor",
            "stockout_penalty_multiplier",
            # Inventory metadata is not a predictive signal; keeping it out avoids
            # accidental target leakage (notably base_daily_demand in demo datasets).
            "base_daily_demand",
            "unit_cost",
            "actual_unit_price",
            "base_unit_price",
            "shelf_life_days",
            "lead_time_days",
            # Real future weather is not known at prediction time unless separately forecast.
            "precpt",
            "avg_temperature",
            "avg_humidity",
            "avg_wind_level",
        }
        self.feature_cols = [
            c
            for c in data.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])
        ]

        logger.info(
            "Transformed feature matrix: %d rows, %d engineered features.",
            len(data),
            len(self.feature_cols),
        )
        return data

    def fit_transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        return self.fit(df).transform(df, drop_na=drop_na)

    def prepare_train_matrices(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Prepares feature matrix X and target vector y for ML models."""
        transformed = self.transform(df, drop_na=True)
        X = transformed[self.feature_cols]
        y = transformed[self.TARGET_COL]
        return X, y, self.feature_cols
