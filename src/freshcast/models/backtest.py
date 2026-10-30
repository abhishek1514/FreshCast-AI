"""Strict Walk-Forward Time-Series Cross Validation (Expanding Window Backtester).

Guarantees zero data leakage by evaluating models on chronologically sequential,
non-overlapping test horizons.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from freshcast.features.pipeline import TimeSeriesFeaturePipeline
from freshcast.metrics.financial import calculate_supply_chain_financial_loss
from freshcast.metrics.standard import calculate_standard_metrics
from freshcast.models.base import BaseDemandForecaster

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Expanding-window time-series backtesting engine."""

    def __init__(
        self,
        n_splits: int = 3,
        test_window_days: int = 28,
        gap_days: int = 0,
    ):
        self.n_splits = n_splits
        self.test_window_days = test_window_days
        self.gap_days = gap_days

    def generate_splits(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generates (train_cutoff, test_start, test_end) for each fold."""
        unique_dates = pd.to_datetime(df[date_col]).drop_duplicates().sort_values()
        max_date = unique_dates.max()

        splits = []
        for i in range(self.n_splits - 1, -1, -1):
            offset_end = i * self.test_window_days
            offset_start = (i + 1) * self.test_window_days

            test_end = max_date - pd.Timedelta(days=offset_end)
            test_start = max_date - pd.Timedelta(days=offset_start - 1)
            train_cutoff = test_start - pd.Timedelta(days=self.gap_days + 1)

            splits.append((train_cutoff, test_start, test_end))

        return splits

    def evaluate_model(
        self,
        model: BaseDemandForecaster,
        df: pd.DataFrame,
        pipeline: TimeSeriesFeaturePipeline,
    ) -> pd.DataFrame:
        """Evaluates a single model across all walk-forward folds."""
        splits = self.generate_splits(df)
        fold_results: list[dict[str, Any]] = []

        logger.info(
            "Starting walk-forward validation for %s across %d folds (window: %d days)...",
            model.name,
            len(splits),
            self.test_window_days,
        )

        for fold_idx, (train_cutoff, test_start, test_end) in enumerate(splits, 1):
            train_raw = df[df["date"] <= train_cutoff].copy()
            test_raw = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

            # Transform features on training partition
            train_transformed = pipeline.fit_transform(train_raw, drop_na=True)
            X_train = train_transformed[pipeline.feature_cols]
            y_train = train_transformed[pipeline.TARGET_COL]

            # Transform test partition with full historical context up to test_end
            combined = pd.concat([train_raw, test_raw], ignore_index=True)
            combined_transformed = pipeline.transform(combined, drop_na=False)

            test_transformed = combined_transformed[
                (combined_transformed["date"] >= test_start)
                & (combined_transformed["date"] <= test_end)
            ].copy()

            X_test = test_transformed[pipeline.feature_cols]
            y_test = test_transformed[pipeline.TARGET_COL]

            # Fit model on training slice
            model.fit(X_train, y_train)

            # Predict on out-of-time test slice
            y_pred = model.predict(X_test)
            test_transformed["pred_sales"] = y_pred

            # Calculate statistical & financial metrics
            std_metrics = calculate_standard_metrics(y_test.values, y_pred)
            fin_metrics = calculate_supply_chain_financial_loss(
                test_transformed,
                pred_col="pred_sales",
                actual_col=pipeline.TARGET_COL,
            )

            res = {
                "model": model.name,
                "fold": fold_idx,
                "train_cutoff": train_cutoff.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "test_cases": int(np.sum(y_test.values)),
                **std_metrics,
                **fin_metrics,
            }
            fold_results.append(res)
            logger.info(
                "Fold %d/%d (%s): WAPE=%.2f%%, MAE=%.1f, FinLoss=$%.2f",
                fold_idx,
                len(splits),
                model.name,
                std_metrics["wape"] * 100,
                std_metrics["mae"],
                fin_metrics["total_financial_loss_usd"],
            )

        return pd.DataFrame(fold_results)

    def benchmark_models(
        self,
        models: list[BaseDemandForecaster],
        df: pd.DataFrame,
        pipeline: TimeSeriesFeaturePipeline,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Benchmarks all candidate models and computes mean aggregate performance."""
        all_fold_dfs: list[pd.DataFrame] = []

        for model in models:
            fold_df = self.evaluate_model(model, df, pipeline)
            all_fold_dfs.append(fold_df)

        full_results = pd.concat(all_fold_dfs, ignore_index=True)

        # Aggregate summary table (mean across folds)
        summary = (
            full_results.groupby("model")
            .agg(
                {
                    "wape": "mean",
                    "mae": "mean",
                    "rmse": "mean",
                    "smape": "mean",
                    "bias": "mean",
                    "total_financial_loss_usd": "mean",
                    "spoilage_loss_usd": "mean",
                    "stockout_loss_usd": "mean",
                    "loss_per_demand_case_usd": "mean",
                }
            )
            .reset_index()
            .sort_values(by="wape")
            .reset_index(drop=True)
        )

        return full_results, summary
