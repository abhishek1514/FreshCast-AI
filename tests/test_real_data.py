import pandas as pd

from scripts.prepare_real_data import choose_series


def test_real_data_adapter_selects_complete_panel():
    rows = []
    for store in (101, 102, 103):
        for product in range(201, 211):
            for date in pd.date_range("2024-01-01", periods=90, freq="D"):
                rows.append({
                    "store_id": store,
                    "product_id": product,
                    "dt": date,
                    "sale_amount": float(product % 7 + 1),
                    "third_category_id": product % 4,
                    "discount": 1.0,
                    "activity_flag": 0,
                    "holiday_flag": 0,
                    "precpt": 0.0,
                    "avg_temperature": 20.0,
                    "avg_humidity": 50.0,
                    "avg_wind_level": 2.0,
                })
    result = choose_series(pd.DataFrame(rows), stores_needed=3, products_needed=10)
    assert result["dc_id"].nunique() == 3
    assert result["sku_id"].nunique() == 10
    assert len(result) == 3 * 10 * 90
    assert not result.duplicated(["date", "dc_id", "sku_id"]).any()
    assert "base_daily_demand" not in result.columns
