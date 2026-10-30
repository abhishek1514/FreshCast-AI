"""Download FreshRetailNet-50K and adapt a small real-data panel for FreshCast.

Source: Dingdong-Inc/FreshRetailNet-50K on Hugging Face, CC BY 4.0.
Important: sale_amount is globally normalized observed sales, not literal cases.
Costs, prices, shelf lives, and lead times are explicitly illustrative assumptions.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("freshcast.real_data")

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/foodservice_daily_sales.csv"
META_PATH = ROOT / "data/raw/real_data_metadata.txt"
DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"


def choose_series(df: pd.DataFrame, stores_needed: int, products_needed: int) -> pd.DataFrame:
    """Pick stores with the most shared complete product series (normally 3 x 10)."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    dates_expected = df["dt"].nunique()
    counts = df.groupby(["store_id", "product_id"])["dt"].nunique()
    complete = counts[counts == dates_expected].reset_index()[["store_id", "product_id"]]
    if complete.empty:
        raise RuntimeError("No complete store-product time series found in downloaded training split.")
    store_products = {
        store: set(group["product_id"].tolist())
        for store, group in complete.groupby("store_id")
    }
    store_rank = sorted(store_products, key=lambda s: len(store_products[s]), reverse=True)[:40]
    best = None
    # Search candidate triples and choose the triple with the most common products.
    import itertools
    for trio in itertools.combinations(store_rank, stores_needed):
        common = set.intersection(*(store_products[s] for s in trio))
        if best is None or len(common) > len(best[1]):
            best = (trio, common)
        if len(common) >= products_needed:
            break
    if best is None or len(best[1]) < products_needed:
        raise RuntimeError(
            f"Could not find {stores_needed} stores sharing {products_needed} complete products. "
            "Try lowering --products or --stores."
        )
    stores, common_products = best
    # Prefer products with highest total observed sales across the selected stores.
    eligible = df[df["store_id"].isin(stores) & df["product_id"].isin(common_products)]
    top_products = (
        eligible.groupby("product_id")["sale_amount"].sum().sort_values(ascending=False)
        .head(products_needed).index.tolist()
    )
    selected = eligible[eligible["product_id"].isin(top_products)].copy()
    selected["dc_id"] = selected["store_id"].map({v: f"DC_REAL_{i+1:02d}" for i, v in enumerate(stores)})
    selected["sku_id"] = selected["product_id"].map({v: f"SKU_REAL_{i+1:02d}" for i, v in enumerate(top_products)})
    selected["date"] = selected["dt"]
    selected["sales_cases"] = pd.to_numeric(selected["sale_amount"], errors="coerce").fillna(0).clip(lower=0)
    selected["category"] = "Category_" + selected["third_category_id"].astype(str)
    selected["dc_name"] = selected["dc_id"].map({f"DC_REAL_{i+1:02d}": f"Observed retail store {int(v)}" for i, v in enumerate(stores)})
    selected["sku_name"] = selected["sku_id"].map({f"SKU_REAL_{i+1:02d}": f"Observed product {int(v)}" for i, v in enumerate(top_products)})
    selected["discount_pct"] = (1.0 - pd.to_numeric(selected["discount"], errors="coerce").fillna(1.0)).clip(0, 1)
    selected["is_promo"] = pd.to_numeric(selected.get("activity_flag", 0), errors="coerce").fillna(0).astype(int)
    selected["holiday_flag"] = pd.to_numeric(selected.get("holiday_flag", 0), errors="coerce").fillna(0).astype(int)
    # Weather values are observed covariates in the source dataset.
    for col in ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]:
        selected[col] = pd.to_numeric(selected.get(col, 0), errors="coerce").fillna(0)
    # FreshRetailNet does not provide item-level cost/price/shelf-life. These are
    # transparent illustrative assumptions for the inventory demo, not source facts.
    selected["unit_cost"] = 1.0
    selected["actual_unit_price"] = 1.30
    selected["base_unit_price"] = 1.30
    selected["shelf_life_days"] = 5
    selected["lead_time_days"] = 2
    selected["spoilage_cost_factor"] = 0.5
    selected["stockout_penalty_multiplier"] = 1.3
    keep = [
        "date", "dc_id", "dc_name", "sku_id", "sku_name", "category", "sales_cases",
        "discount_pct", "is_promo", "holiday_flag", "precpt", "avg_temperature",
        "avg_humidity", "avg_wind_level", "unit_cost", "actual_unit_price",
        "base_unit_price", "shelf_life_days", "lead_time_days", "spoilage_cost_factor",
        "stockout_penalty_multiplier",
    ]
    result = selected[keep].drop_duplicates(["date", "dc_id", "sku_id"])
    result = result.sort_values(["dc_id", "sku_id", "date"]).reset_index(drop=True)
    # Sanity-check complete daily panel.
    expected = result.groupby(["dc_id", "sku_id"])["date"].nunique().min()
    if result.duplicated(["date", "dc_id", "sku_id"]).any():
        raise RuntimeError("Duplicate date/store/product rows after conversion.")
    LOG.info("Prepared %s rows, %s stores, %s products, %s dates per series", len(result), result.dc_id.nunique(), result.sku_id.nunique(), expected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", type=int, default=3)
    parser.add_argument("--products", type=int, default=10)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install -e .") from exc
    LOG.info("Downloading/reading %s split=%s from Hugging Face", DATASET_ID, args.split)
    dataset = load_dataset(DATASET_ID, split=args.split)
    # Drop nested hourly arrays before materializing in pandas to keep RAM use manageable.
    removable = [c for c in ["hours_sale", "hours_stock_status"] if c in dataset.column_names]
    if removable:
        dataset = dataset.remove_columns(removable)
    frame = dataset.to_pandas()
    LOG.info("Loaded source data: %s rows; columns=%s", len(frame), list(frame.columns))
    prepared = choose_series(frame, args.stores, args.products)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(RAW_PATH, index=False)
    META_PATH.write_text(
        "DATA SOURCE: FreshRetailNet-50K, Dingdong-Inc, Hugging Face.\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "Dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K\n"
        "Paper: https://arxiv.org/abs/2505.16319\n"
        "Observed target: sale_amount, a globally normalized sales measure; not literal cases.\n"
        "Observed covariates: discount, activity/holiday flags, precipitation, temperature, humidity, wind.\n"
        "IMPORTANT: source dataset does not provide actual item costs, prices, shelf lives, or lead times.\n"
        "Inventory financial estimates in this demo use illustrative assumptions (unit_cost=1.00, price=1.30,\n"
        "shelf_life=5 days, lead_time=2 days, spoilage_factor=0.5, stockout_multiplier=1.3).\n"
        "Do not represent those assumptions as real supplier financials.\n",
        encoding="utf-8",
    )
    LOG.info("Wrote %s", RAW_PATH)
    LOG.info("Attribution/assumptions recorded in %s", META_PATH)

if __name__ == "__main__":
    main()
