"""Supply chain financial and asymmetric loss evaluation metrics.

Simulates bottom-line P&L impact for regional foodservice distribution centers:
- Over-stocking cost = Spoilage of perishable inventory
- Under-stocking cost = Lost profit margin + Customer service contract breach penalty
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_supply_chain_financial_loss(
    df: pd.DataFrame,
    pred_col: str = "pred_sales",
    actual_col: str = "sales_cases",
    unit_cost_col: str = "unit_cost",
    unit_price_col: str = "actual_unit_price",
    spoilage_factor_col: str = "spoilage_cost_factor",
    stockout_multiplier_col: str = "stockout_penalty_multiplier",
) -> dict[str, float]:
    """Calculates asymmetric monetary loss for over-forecasting vs under-forecasting."""
    y_true = df[actual_col].to_numpy()
    y_pred = np.clip(df[pred_col].to_numpy(), 0, None)
    unit_cost = df[unit_cost_col].to_numpy()
    unit_price = df[unit_price_col].to_numpy() if unit_price_col in df.columns else unit_cost * 1.30
    margin = np.maximum(unit_price - unit_cost, 0.0)

    spoilage_factors = (
        df[spoilage_factor_col].to_numpy()
        if spoilage_factor_col in df.columns
        else np.ones_like(y_true)
    )
    stockout_multipliers = (
        df[stockout_multiplier_col].to_numpy()
        if stockout_multiplier_col in df.columns
        else np.full_like(y_true, 1.3)
    )

    error = y_pred - y_true
    # Over-forecasting error (excess inventory)
    overstock_units = np.maximum(error, 0.0)
    # Under-forecasting error (unfulfilled customer orders)
    understock_units = np.maximum(-error, 0.0)

    # Cost of spoilage: excess units * unit acquisition cost * perishability risk
    spoilage_cost = np.sum(overstock_units * unit_cost * spoilage_factors)

    # Cost of stockout: lost margin * penalty factor (lost customer goodwill & expedited shipping)
    stockout_cost = np.sum(understock_units * margin * stockout_multipliers)

    total_financial_loss = spoilage_cost + stockout_cost
    total_actual_cases = np.sum(y_true)
    loss_per_case = total_financial_loss / total_actual_cases if total_actual_cases > 0 else 0.0

    return {
        "total_financial_loss_usd": float(total_financial_loss),
        "spoilage_loss_usd": float(spoilage_cost),
        "stockout_loss_usd": float(stockout_cost),
        "loss_per_demand_case_usd": float(loss_per_case),
        "total_overstock_cases": float(np.sum(overstock_units)),
        "total_understock_cases": float(np.sum(understock_units)),
    }
