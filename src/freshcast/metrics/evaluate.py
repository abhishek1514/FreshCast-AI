"""Business and Financial Evaluation Reporting for FreshCast."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from freshcast.metrics.financial import calculate_supply_chain_financial_loss
from freshcast.metrics.standard import calculate_standard_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_report(
    bundle_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generates granular category-level and SKU-level performance reports."""
    if bundle_path is None:
        bundle_path = Path("models/champion_model.joblib")
    if output_dir is None:
        output_dir = Path("data/processed")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(bundle_path)
    eval_df = bundle.get("evaluation_df")
    if eval_df is None:
        raise ValueError("evaluation_df not found in champion bundle.")

    # 1. Performance by Category
    category_rows = []
    for category, group in eval_df.groupby("category"):
        std = calculate_standard_metrics(group["sales_cases"].values, group["pred_sales"].values)
        fin = calculate_supply_chain_financial_loss(
            group, pred_col="pred_sales", actual_col="sales_cases"
        )
        category_rows.append(
            {
                "category": category,
                "total_demand_cases": int(group["sales_cases"].sum()),
                "wape_pct": round(std["wape"] * 100, 2),
                "mae": round(std["mae"], 2),
                "spoilage_loss_usd": round(fin["spoilage_loss_usd"], 2),
                "stockout_loss_usd": round(fin["stockout_loss_usd"], 2),
                "total_financial_loss_usd": round(fin["total_financial_loss_usd"], 2),
                "loss_per_case_usd": round(fin["loss_per_demand_case_usd"], 2),
            }
        )
    cat_df = pd.DataFrame(category_rows).sort_values(by="total_financial_loss_usd", ascending=False)
    cat_df.to_csv(out_path / "category_evaluation_report.csv", index=False)

    # 3. Performance by SKU
    sku_rows = []
    for (sku_id, sku_name, category), group in eval_df.groupby(["sku_id", "sku_name", "category"]):
        std = calculate_standard_metrics(group["sales_cases"].values, group["pred_sales"].values)
        fin = calculate_supply_chain_financial_loss(
            group, pred_col="pred_sales", actual_col="sales_cases"
        )
        sku_rows.append(
            {
                "sku_id": sku_id,
                "sku_name": sku_name,
                "category": category,
                "shelf_life_days": group["shelf_life_days"].iloc[0],
                "total_demand_cases": int(group["sales_cases"].sum()),
                "wape_pct": round(std["wape"] * 100, 2),
                "mae": round(std["mae"], 2),
                "spoilage_loss_usd": round(fin["spoilage_loss_usd"], 2),
                "stockout_loss_usd": round(fin["stockout_loss_usd"], 2),
                "total_financial_loss_usd": round(fin["total_financial_loss_usd"], 2),
            }
        )
    sku_df = pd.DataFrame(sku_rows).sort_values(by="total_financial_loss_usd", ascending=False)
    sku_df.to_csv(out_path / "sku_evaluation_report.csv", index=False)

    print("\n" + "=" * 80)
    print("FRESHCAST CATEGORY-LEVEL EVALUATION REPORT")
    print("=" * 80)
    print(cat_df.to_string(index=False))
    print("=" * 80 + "\n")

    return {"category": cat_df, "sku": sku_df}


if __name__ == "__main__":
    generate_evaluation_report()
