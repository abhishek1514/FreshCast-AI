"""Realistic Foodservice Distribution Sales Data Generator.

Simulates multi-SKU daily sales volumes across Regional Distribution Centers (RDCs)
mirroring enterprise foodservice supply chain patterns:
- Restaurant ordering cycles (prep days for weekend dining)
- Category-specific annual seasonality
- US National Holiday demand surges
- Promotional price elasticity
- Perishable shelf-life constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SKUConfig:
    id: str
    name: str
    category: str
    shelf_life_days: int
    lead_time_days: int
    unit_cost: float
    unit_price: float
    spoilage_cost_factor: float
    stockout_penalty_multiplier: float
    base_daily_demand: float


@dataclass
class DCConfig:
    id: str
    name: str
    capacity_cases: int
    base_volume_multiplier: float


class FoodserviceDataGenerator:
    """Generates synthetic multi-SKU daily sales history for foodservice distribution centers."""

    # Weekly multiplier: Day 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    # Foodservice delivery: Sun night / Mon morning delivery is huge for weekly restocking.
    # Wed/Thu delivery is huge for weekend restaurant prep. Sat/Sun deliveries are lighter.
    WEEKLY_FACTORS = {
        0: 1.30,  # Monday: heavy restocking after weekend
        1: 0.95,  # Tuesday: moderate
        2: 1.15,  # Wednesday: prep for weekend dining
        3: 1.35,  # Thursday: peak restaurant prep delivery
        4: 1.10,  # Friday: last-minute weekend supply
        5: 0.60,  # Saturday: skeleton warehouse shifts
        6: 0.55,  # Sunday: warehouse prep, low deliveries
    }

    US_HOLIDAYS = [
        # (Month, Day, Name, WindowDaysBefore, Multiplier)
        (1, 1, "New Year's Day", 3, 1.25),
        (2, 14, "Valentine's Day", 3, 1.35),
        (5, 12, "Mother's Day", 4, 1.45),  # Massive restaurant dining day
        (5, 27, "Memorial Day", 5, 1.30),
        (7, 4, "Independence Day", 4, 1.35),
        (9, 2, "Labor Day", 4, 1.25),
        (11, 28, "Thanksgiving", 7, 1.60),  # Heavy banquet / catering orders
        (12, 24, "Christmas Eve", 6, 1.50),
        (12, 31, "New Year's Eve", 4, 1.40),
    ]

    def __init__(self, config_path: str | Path | None = None):
        self.config = self._load_config(config_path)
        self.random_seed = self.config.get("project", {}).get("random_seed", 42)
        np.random.seed(self.random_seed)

        self.dcs: list[DCConfig] = [
            DCConfig(**dc) for dc in self.config.get("distribution_centers", [])
        ]
        self.skus: list[SKUConfig] = [SKUConfig(**sku) for sku in self.config.get("skus", [])]
        self.start_date = self.config.get("data", {}).get("start_date", "2022-01-01")
        self.end_date = self.config.get("data", {}).get("end_date", "2024-03-31")

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            # Look in standard repo location
            candidate = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
            if candidate.exists():
                config_path = candidate
            else:
                config_path = Path("configs/default_config.yaml")

        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
        logger.warning("Config file not found at %s. Using default fallback configuration.", path)
        return self._default_fallback_config()

    def _default_fallback_config(self) -> dict[str, Any]:
        return {
            "project": {"random_seed": 42},
            "data": {
                "start_date": "2022-01-01",
                "end_date": "2024-03-31",
                "raw_dir": "data/raw",
            },
            "distribution_centers": [
                {
                    "id": "DC_BOS",
                    "name": "Northeast DC (Boston)",
                    "capacity_cases": 500000,
                    "base_volume_multiplier": 1.15,
                },
                {
                    "id": "DC_ATL",
                    "name": "Southeast DC (Atlanta)",
                    "capacity_cases": 650000,
                    "base_volume_multiplier": 1.25,
                },
                {
                    "id": "DC_CHI",
                    "name": "Midwest DC (Chicago)",
                    "capacity_cases": 750000,
                    "base_volume_multiplier": 1.40,
                },
            ],
            "skus": [
                {
                    "id": "SKU_BEEF_SIRLOIN",
                    "name": "Choice Beef Sirloin",
                    "category": "Meat & Poultry",
                    "shelf_life_days": 6,
                    "lead_time_days": 2,
                    "unit_cost": 85.0,
                    "unit_price": 110.0,
                    "spoilage_cost_factor": 1.0,
                    "stockout_penalty_multiplier": 1.5,
                    "base_daily_demand": 120,
                },
                {
                    "id": "SKU_CHICKEN_BREAST",
                    "name": "Chicken Breast",
                    "category": "Meat & Poultry",
                    "shelf_life_days": 5,
                    "lead_time_days": 2,
                    "unit_cost": 62.0,
                    "unit_price": 82.0,
                    "spoilage_cost_factor": 1.0,
                    "stockout_penalty_multiplier": 1.4,
                    "base_daily_demand": 240,
                },
                {
                    "id": "SKU_SALMON_FILLET",
                    "name": "Atlantic Salmon",
                    "category": "Seafood",
                    "shelf_life_days": 4,
                    "lead_time_days": 2,
                    "unit_cost": 95.0,
                    "unit_price": 130.0,
                    "spoilage_cost_factor": 1.0,
                    "stockout_penalty_multiplier": 1.6,
                    "base_daily_demand": 90,
                },
                {
                    "id": "SKU_ROMAINE_HEARTS",
                    "name": "Romaine Hearts",
                    "category": "Produce",
                    "shelf_life_days": 5,
                    "lead_time_days": 1,
                    "unit_cost": 18.0,
                    "unit_price": 28.0,
                    "spoilage_cost_factor": 1.0,
                    "stockout_penalty_multiplier": 1.3,
                    "base_daily_demand": 320,
                },
                {
                    "id": "SKU_ROMA_TOMATOES",
                    "name": "Roma Tomatoes",
                    "category": "Produce",
                    "shelf_life_days": 6,
                    "lead_time_days": 1,
                    "unit_cost": 22.0,
                    "unit_price": 34.0,
                    "spoilage_cost_factor": 1.0,
                    "stockout_penalty_multiplier": 1.3,
                    "base_daily_demand": 280,
                },
                {
                    "id": "SKU_AVOCADO_HASS",
                    "name": "Hass Avocados",
                    "category": "Produce",
                    "shelf_life_days": 5,
                    "lead_time_days": 2,
                    "unit_cost": 38.0,
                    "unit_price": 54.0,
                    "spoilage_cost_factor": 0.9,
                    "stockout_penalty_multiplier": 1.4,
                    "base_daily_demand": 160,
                },
                {
                    "id": "SKU_HEAVY_CREAM",
                    "name": "Heavy Cream 36%",
                    "category": "Dairy",
                    "shelf_life_days": 14,
                    "lead_time_days": 2,
                    "unit_cost": 32.0,
                    "unit_price": 45.0,
                    "spoilage_cost_factor": 0.8,
                    "stockout_penalty_multiplier": 1.25,
                    "base_daily_demand": 180,
                },
                {
                    "id": "SKU_CHEDDAR_CHEESE",
                    "name": "White Cheddar",
                    "category": "Dairy",
                    "shelf_life_days": 28,
                    "lead_time_days": 3,
                    "unit_cost": 42.0,
                    "unit_price": 58.0,
                    "spoilage_cost_factor": 0.5,
                    "stockout_penalty_multiplier": 1.2,
                    "base_daily_demand": 150,
                },
                {
                    "id": "SKU_FRENCH_FRIES",
                    "name": "Thin French Fries",
                    "category": "Frozen",
                    "shelf_life_days": 180,
                    "lead_time_days": 4,
                    "unit_cost": 24.0,
                    "unit_price": 36.0,
                    "spoilage_cost_factor": 0.1,
                    "stockout_penalty_multiplier": 1.2,
                    "base_daily_demand": 450,
                },
                {
                    "id": "SKU_CANOLA_OIL",
                    "name": "Canola Frying Oil",
                    "category": "Dry & Pantry",
                    "shelf_life_days": 240,
                    "lead_time_days": 5,
                    "unit_cost": 28.0,
                    "unit_price": 40.0,
                    "spoilage_cost_factor": 0.05,
                    "stockout_penalty_multiplier": 1.2,
                    "base_daily_demand": 210,
                },
            ],
        }

    def _calculate_seasonal_factor(self, date: pd.Timestamp, category: str) -> float:
        """Category-specific annual sinusoidal seasonality."""
        day_of_year = date.day_of_year
        # Produce & Meats peak in summer (grilling / fresh patio dining)
        if category in ["Produce", "Meat & Poultry", "Seafood"]:
            # Peak around day 190 (mid-July)
            season = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
        elif category == "Dairy":
            # Peak in Q4 holiday cooking & baking
            season = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 280) / 365.25)
        elif category == "Frozen":
            # Consistent demand with minor summer lift
            season = 1.0 + 0.08 * np.sin(2 * np.pi * (day_of_year - 150) / 365.25)
        else:
            season = 1.0 + 0.05 * np.sin(2 * np.pi * day_of_year / 365.25)
        return float(season)

    def _calculate_holiday_multiplier(self, date: pd.Timestamp) -> float:
        """Checks proximity to major restaurant surge holidays."""
        multiplier = 1.0
        for m, d, _, window, surge in self.US_HOLIDAYS:
            # Check for current year and surrounding years
            try:
                h_date = pd.Timestamp(year=date.year, month=m, day=d)
            except ValueError:
                continue
            days_diff = (h_date - date).days
            if 0 <= days_diff <= window:
                # Ramp up as holiday approaches
                proximity_weight = (window - days_diff + 1) / (window + 1)
                item_surge = 1.0 + (surge - 1.0) * proximity_weight
                multiplier = max(multiplier, item_surge)
        return multiplier

    def generate(self) -> pd.DataFrame:
        """Generates the full panel dataset across dates, distribution centers, and SKUs."""
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        n_days = len(date_range)
        logger.info(
            "Generating %d daily timestamps from %s to %s",
            n_days,
            self.start_date,
            self.end_date,
        )

        records: list[dict[str, Any]] = []

        # Overall macroeconomic linear trend (e.g. 3% annual business growth)
        trend = np.linspace(1.0, 1.07, n_days)

        for dc in self.dcs:
            for sku in self.skus:
                # Promotion schedule: generate realistic 4-day to 7-day promotional cycles
                # roughly every 45-60 days
                promo_active = np.zeros(n_days, dtype=bool)
                promo_discount = np.zeros(n_days, dtype=float)

                t = 20
                while t < n_days - 10:
                    gap = np.random.randint(35, 65)
                    t += gap
                    if t >= n_days:
                        break
                    duration = np.random.randint(4, 8)
                    end_t = min(t + duration, n_days)
                    discount = np.random.choice([0.10, 0.15, 0.20, 0.25])
                    promo_active[t:end_t] = True
                    promo_discount[t:end_t] = discount
                    t = end_t

                for i, date in enumerate(date_range):
                    day_of_week = date.day_of_week
                    weekly_factor = self.WEEKLY_FACTORS[day_of_week]
                    seasonal_factor = self._calculate_seasonal_factor(date, sku.category)
                    holiday_factor = self._calculate_holiday_multiplier(date)
                    current_trend = trend[i]

                    # Promotional price elasticity: elasticity ~ -1.6
                    price_discount = promo_discount[i]
                    promo_multiplier = 1.0 + (1.6 * price_discount) if promo_active[i] else 1.0
                    actual_unit_price = round(sku.unit_price * (1.0 - price_discount), 2)

                    # Expected mean order volume
                    expected_demand = (
                        sku.base_daily_demand
                        * dc.base_volume_multiplier
                        * weekly_factor
                        * seasonal_factor
                        * holiday_factor
                        * current_trend
                        * promo_multiplier
                    )

                    # Stochastic variation using Negative Binomial to capture real retail overdispersion
                    # Variance = mean + alpha * mean^2
                    alpha = 0.05
                    var = expected_demand + alpha * (expected_demand**2)
                    p = expected_demand / var
                    n = (expected_demand**2) / (var - expected_demand)
                    actual_sales = int(np.random.negative_binomial(n, p))
                    # Ensure positive sales
                    actual_sales = max(0, actual_sales)

                    records.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "dc_id": dc.id,
                            "dc_name": dc.name,
                            "sku_id": sku.id,
                            "sku_name": sku.name,
                            "category": sku.category,
                            "shelf_life_days": sku.shelf_life_days,
                            "lead_time_days": sku.lead_time_days,
                            "unit_cost": sku.unit_cost,
                            "base_unit_price": sku.unit_price,
                            "actual_unit_price": actual_unit_price,
                            "discount_pct": round(price_discount, 2),
                            "is_promo": int(promo_active[i]),
                            "spoilage_cost_factor": sku.spoilage_cost_factor,
                            "stockout_penalty_multiplier": sku.stockout_penalty_multiplier,
                            "sales_cases": actual_sales,
                        }
                    )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["dc_id", "sku_id", "date"]).reset_index(drop=True)

        logger.info(
            "Generated dataset with %d rows across %d distribution centers and %d SKUs.",
            len(df),
            len(self.dcs),
            len(self.skus),
        )
        return df

    def save(self, df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
        """Saves generated dataset to raw CSV path."""
        if output_path is None:
            raw_dir = Path(self.config.get("data", {}).get("raw_dir", "data/raw"))
            output_path = raw_dir / "foodservice_daily_sales.csv"

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved foodservice sales dataset to %s", out)
        return out


def main():
    generator = FoodserviceDataGenerator()
    df = generator.generate()
    saved_path = generator.save(df)
    print(f"Data generation complete! Saved to {saved_path}. Shape: {df.shape}")


if __name__ == "__main__":
    main()

