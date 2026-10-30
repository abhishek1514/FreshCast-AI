"""Autoregressive lag and rolling window feature extraction."""

from __future__ import annotations

import pandas as pd


class LagFeatureExtractor:
    """Computes autoregressive lags and rolling statistics per DC-SKU series.

    Strictly enforces zero lookahead bias by shifting target values prior to
    computing rolling metrics.
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        target_col: str = "sales_cases",
        group_cols: list[str] | None = None,
    ):
        self.lags = lags if lags is not None else [1, 2, 3, 7, 14, 21, 28]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [7, 14, 28]
        self.target_col = target_col
        self.group_cols = group_cols if group_cols is not None else ["dc_id", "sku_id"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by=self.group_cols + ["date"]).reset_index(drop=True)

        grouped = df.groupby(self.group_cols)[self.target_col]

        # 1. Discrete autoregressive lag features
        for lag in self.lags:
            df[f"lag_{lag}"] = grouped.shift(lag)

        # 2. Rolling window aggregations on shifted series (lag_1) to prevent target leakage
        for window in self.rolling_windows:
            # Group again on the shifted series to respect DC-SKU boundaries
            shifted_grouped = df.groupby(self.group_cols)["lag_1"]
            rolling_obj = shifted_grouped.rolling(
                window=window, min_periods=max(1, min(window, window // 2))
            )

            df[f"rolling_mean_{window}"] = rolling_obj.mean().reset_index(
                level=list(range(len(self.group_cols))), drop=True
            )
            df[f"rolling_std_{window}"] = rolling_obj.std().reset_index(
                level=list(range(len(self.group_cols))), drop=True
            )
            df[f"rolling_min_{window}"] = rolling_obj.min().reset_index(
                level=list(range(len(self.group_cols))), drop=True
            )
            df[f"rolling_max_{window}"] = rolling_obj.max().reset_index(
                level=list(range(len(self.group_cols))), drop=True
            )

        # 4. Momentum & Relative Demand Ratios
        eps = 1e-5
        if "rolling_mean_7" in df.columns:
            df["demand_momentum_7d"] = df["lag_1"] / (df["rolling_mean_7"] + eps)
        if "rolling_mean_7" in df.columns and "rolling_mean_28" in df.columns:
            df["demand_trend_ratio"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + eps)

        # 5. Volatility coefficient of variation (CV = std / mean)
        if "rolling_mean_7" in df.columns and "rolling_std_7" in df.columns:
            df["demand_volatility_7d"] = df["rolling_std_7"] / (df["rolling_mean_7"] + eps)

        return df
