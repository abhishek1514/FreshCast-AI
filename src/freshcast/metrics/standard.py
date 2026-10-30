"""Standard statistical metrics for time-series demand forecasting."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE = mean(|y - y_hat|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = sqrt(mean((y - y_hat)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE = sum(|y - y_hat|) / sum(y).

    Gold-standard metric in retail & supply chain forecasting because it avoids
    infinite or exploded MAPE values on low-volume / zero-demand days.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE = 100 * mean( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast Bias = sum(y_hat - y) / sum(y).

    Positive indicates persistent over-forecasting (spoilage risk);
    Negative indicates persistent under-forecasting (stockout risk).
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(y_pred - y_true) / total_actual)


def calculate_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes full suite of standard forecast evaluation metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    return {
        "wape": weighted_absolute_percentage_error(y_t, y_p),
        "mae": mean_absolute_error(y_t, y_p),
        "rmse": root_mean_squared_error(y_t, y_p),
        "smape": symmetric_mean_absolute_percentage_error(y_t, y_p),
        "bias": forecast_bias(y_t, y_p),
    }
