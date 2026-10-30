"""Base forecaster interface for FreshCast."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseDemandForecaster(ABC):
    """Abstract base class for all FreshCast demand forecasting models."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseDemandForecaster:
        """Fits the model to training features and target."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for feature matrix X."""

    def save(self, filepath: str | Path) -> None:
        """Serializes model to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> BaseDemandForecaster:
        """Loads serialized model from disk."""
        return joblib.load(filepath)
