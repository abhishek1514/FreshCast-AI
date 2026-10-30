"""Unit tests for statistical and financial supply chain metrics."""

import numpy as np
import pandas as pd

from freshcast.metrics.financial import calculate_supply_chain_financial_loss
from freshcast.metrics.standard import (
    forecast_bias,
    mean_absolute_error,
    weighted_absolute_percentage_error,
)


def test_standard_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 20.0 / 3.0)

    # sum(|e|) = 10 + 10 + 0 = 20, sum(y) = 600 -> WAPE = 20 / 600 = 0.0333...
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(wape, 20.0 / 600.0)

    # bias = (10 - 10 + 0) / 600 = 0
    bias = forecast_bias(y_true, y_pred)
    assert np.isclose(bias, 0.0)


def test_asymmetric_financial_loss():
    # Case 1: Over-forecast by 10 units of $50 cost meat item with 1.0 spoilage factor
    # Case 2: Under-forecast by 10 units with margin = $30 ($100 price - $70 cost) and penalty = 1.5
    df = pd.DataFrame(
        {
            "sales_cases": [100.0, 100.0],
            "pred_sales": [110.0, 90.0],
            "unit_cost": [50.0, 70.0],
            "actual_unit_price": [75.0, 100.0],
            "spoilage_cost_factor": [1.0, 1.0],
            "stockout_penalty_multiplier": [1.5, 1.5],
        }
    )

    fin = calculate_supply_chain_financial_loss(
        df,
        pred_col="pred_sales",
        actual_col="sales_cases",
    )

    # Row 1: Overstock = 10 * 50 * 1.0 = $500 spoilage
    # Row 2: Understock = 10 * (100 - 70) * 1.5 = $450 lost margin penalty
    assert np.isclose(fin["spoilage_loss_usd"], 500.0)
    assert np.isclose(fin["stockout_loss_usd"], 450.0)
    assert np.isclose(fin["total_financial_loss_usd"], 950.0)
