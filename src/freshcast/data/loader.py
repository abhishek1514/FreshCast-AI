"""Data loading, validation, and preparation module for FreshCast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading, cleaning, and validating time series data for FreshCast."""

    REQUIRED_COLUMNS = [
        "date",
        "dc_id",
        "sku_id",
        "category",
        "sales_cases",
    ]

    def __init__(self, config_path: str | Path | None = None):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            candidate = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
            if candidate.exists():
                config_path = candidate
            else:
                config_path = Path("configs/default_config.yaml")

        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def load_raw_data(self, file_path: str | Path | None = None) -> pd.DataFrame:
        """Loads raw foodservice sales CSV and ensures date typing and ordering."""
        if file_path is None:
            raw_dir = Path(self.config.get("data", {}).get("raw_dir", "data/raw"))
            file_path = raw_dir / "foodservice_daily_sales.csv"

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found at {path}. Run FoodserviceDataGenerator first."
            )

        df = pd.read_csv(path)
        return self.validate_and_clean(df)

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates schema, converts dates, and verifies panel continuity."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Enforce non-negative sales
        df["sales_cases"] = df["sales_cases"].clip(lower=0)

        # Sort explicitly
        df = df.sort_values(by=["dc_id", "sku_id", "date"]).reset_index(drop=True)

        # Ensure complete grid (no missing dates in time-series)
        complete_dfs: list[pd.DataFrame] = []
        for (dc, sku), group in df.groupby(["dc_id", "sku_id"]):
            min_date = group["date"].min()
            max_date = group["date"].max()
            full_dates = pd.date_range(min_date, max_date, freq="D", name="date")
            group_reindexed = group.set_index("date").reindex(full_dates)

            # Fill missing forward/backward static attributes, fill missing sales with 0
            group_reindexed["dc_id"] = dc
            group_reindexed["sku_id"] = sku
            group_reindexed["sales_cases"] = group_reindexed["sales_cases"].fillna(0)
            group_reindexed = group_reindexed.ffill().bfill()
            group_reindexed = group_reindexed.reset_index().rename(columns={"index": "date"})
            complete_dfs.append(group_reindexed)

        full_df = pd.concat(complete_dfs, ignore_index=True)
        full_df = full_df.sort_values(by=["dc_id", "sku_id", "date"]).reset_index(drop=True)

        logger.info(
            "Validated dataset: %d rows across %d series.",
            len(full_df),
            len(complete_dfs),
        )
        return full_df

    def split_train_test_by_date(
        self, df: pd.DataFrame, test_days: int = 28
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strict temporal split: hold out the last `test_days` for out-of-time evaluation."""
        max_date = df["date"].max()
        cutoff_date = max_date - pd.Timedelta(days=test_days - 1)

        train_df = df[df["date"] < cutoff_date].copy().reset_index(drop=True)
        test_df = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)

        logger.info(
            "Temporal Train/Test Split: Train up to %s (%d rows), Test from %s to %s (%d rows)",
            cutoff_date - pd.Timedelta(days=1),
            len(train_df),
            test_df["date"].min().strftime("%Y-%m-%d"),
            test_df["date"].max().strftime("%Y-%m-%d"),
            len(test_df),
        )
        return train_df, test_df
