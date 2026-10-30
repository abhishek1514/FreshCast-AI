"""Production FastAPI service for FreshCast demand forecasting and replenishment."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from freshcast.api.schemas import (
    DailyForecast,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReplenishmentRequest,
    ReplenishmentResponse,
)

logger = logging.getLogger("freshcast.api")
logging.basicConfig(level=logging.INFO)

# Global in-memory cache for model bundle and historical records
model_bundle: dict[str, Any] = {}
historical_df: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads champion model bundle and data panel upon service startup."""
    global model_bundle, historical_df
    bundle_path = Path("models/champion_model.joblib")
    data_path = Path("data/raw/foodservice_daily_sales.csv")

    if bundle_path.exists():
        logger.info("Loading champion model bundle from %s...", bundle_path)
        model_bundle = joblib.load(bundle_path)
    else:
        logger.warning(
            "Champion model bundle not found at %s. Please run training pipeline.",
            bundle_path,
        )

    if data_path.exists():
        logger.info("Loading historical panel from %s...", data_path)
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        historical_df = df
    else:
        logger.warning("Historical data not found at %s.", data_path)

    yield
    model_bundle.clear()


app = FastAPI(
    title="FreshCast Demand Forecasting & Replenishment API",
    description="Enterprise API providing multi-SKU commercial foodservice demand predictions, safety stock calculation, and promotional what-if simulation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health status, model availability, and active catalog scope."""
    is_loaded = bool(model_bundle and "model" in model_bundle)
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=is_loaded,
        active_dcs=3,
        active_skus=10,
    )


@app.get("/api/v1/metadata", tags=["Metadata"])
def get_catalog_metadata():
    """Returns available distribution centers and SKU product definitions."""
    if historical_df is None:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded.")

    dcs = historical_df[["dc_id", "dc_name"]].drop_duplicates().to_dict(orient="records")
    skus = (
        historical_df[
            [
                "sku_id",
                "sku_name",
                "category",
                "shelf_life_days",
                "lead_time_days",
                "unit_cost",
                "base_unit_price",
            ]
        ]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    return {"distribution_centers": dcs, "skus": skus}


@app.post("/api/v1/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def generate_forecast(req: ForecastRequest):
    """Generates multi-day demand predictions with confidence bounds and promotional simulation."""
    if not model_bundle or "model" not in model_bundle or "pipeline" not in model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded.",
        )
    if historical_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical sales panel not loaded.",
        )

    # Filter for series
    series = (
        historical_df[
            (historical_df["dc_id"] == req.dc_id) & (historical_df["sku_id"] == req.sku_id)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if series.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No time-series found for DC '{req.dc_id}' and SKU '{req.sku_id}'.",
        )

    meta = series.iloc[-1]
    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    last_date = series["date"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=req.horizon_days, freq="D"
    )

    # Construct simulation frame
    future_rows = []
    base_price = meta["base_unit_price"]
    act_price = round(base_price * (1.0 - req.discount_pct), 2)
    is_promo = 1 if req.discount_pct > 0 else 0

    for f_date in future_dates:
        future_rows.append(
            {
                "date": f_date,
                "dc_id": req.dc_id,
                "dc_name": meta["dc_name"],
                "sku_id": req.sku_id,
                "sku_name": meta["sku_name"],
                "category": meta["category"],
                "shelf_life_days": meta["shelf_life_days"],
                "lead_time_days": meta["lead_time_days"],
                "unit_cost": meta["unit_cost"],
                "base_unit_price": base_price,
                "actual_unit_price": act_price,
                "discount_pct": req.discount_pct,
                "is_promo": is_promo,
                "sales_cases": 0.0,
            }
        )

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([series, future_df], ignore_index=True)
    transformed = pipeline.transform(combined, drop_na=False)

    future_transformed = transformed[transformed["date"].isin(future_dates)].copy()
    X_future = future_transformed[pipeline.feature_cols]

    preds = model.predict(X_future)

    # Apply promo elasticity lift if discount specified
    if req.discount_pct > 0:
        elasticity_factor = 1.0 + (1.6 * req.discount_pct)
        preds = preds * elasticity_factor

    recent_std = series["sales_cases"].tail(28).std()
    recent_std = recent_std if not np.isnan(recent_std) and recent_std > 0 else 10.0

    daily_forecasts: list[DailyForecast] = []
    for i, f_date in enumerate(future_dates):
        p = int(round(preds[i]))
        lower_bound = max(0, int(round(p - (1.645 * recent_std))))
        upper_bound = int(round(p + (1.645 * recent_std)))
        daily_forecasts.append(
            DailyForecast(
                date=f_date.strftime("%Y-%m-%d"),
                forecasted_demand=p,
                lower_bound_p10=lower_bound,
                upper_bound_p90=upper_bound,
                unit_price=act_price,
                is_promo=bool(is_promo),
            )
        )

    total_demand = sum(d.forecasted_demand for d in daily_forecasts)
    avg_demand = round(total_demand / req.horizon_days, 2)

    return ForecastResponse(
        dc_id=req.dc_id,
        dc_name=meta["dc_name"],
        sku_id=req.sku_id,
        sku_name=meta["sku_name"],
        category=meta["category"],
        horizon_days=req.horizon_days,
        total_projected_demand=total_demand,
        average_daily_demand=avg_demand,
        daily_forecasts=daily_forecasts,
    )


@app.post(
    "/api/v1/replenish",
    response_model=ReplenishmentResponse,
    tags=["Inventory Replenishment"],
)
def compute_replenishment(req: ReplenishmentRequest):
    """Calculates Safety Stock and Reorder Point (ROP) for warehouse inventory optimization."""
    if historical_df is None:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded.")

    series = (
        historical_df[
            (historical_df["dc_id"] == req.dc_id) & (historical_df["sku_id"] == req.sku_id)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Series not found for DC '{req.dc_id}' and SKU '{req.sku_id}'.",
        )

    meta = series.iloc[-1]
    lead_time = int(meta["lead_time_days"])
    shelf_life = int(meta["shelf_life_days"])
    unit_cost = float(meta["unit_cost"])

    recent_sales = series["sales_cases"].tail(28)
    avg_daily_demand = float(recent_sales.mean())
    daily_sigma = float(recent_sales.std())
    daily_sigma = daily_sigma if not np.isnan(daily_sigma) and daily_sigma > 0 else 5.0

    # Z-factor for service level
    z_factors = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054}
    z_score = z_factors.get(req.service_level, 1.645)

    lead_time_demand = round(avg_daily_demand * lead_time, 2)
    lead_time_sigma = daily_sigma * np.sqrt(lead_time)
    safety_stock = int(np.ceil(z_score * lead_time_sigma))
    reorder_point = int(np.ceil(lead_time_demand + safety_stock))

    reorder_recommended = False
    if req.current_on_hand_inventory is not None:
        reorder_recommended = req.current_on_hand_inventory <= reorder_point

    spoilage_alert = shelf_life <= 6

    return ReplenishmentResponse(
        dc_id=req.dc_id,
        sku_id=req.sku_id,
        sku_name=meta["sku_name"],
        category=meta["category"],
        shelf_life_days=shelf_life,
        lead_time_days=lead_time,
        average_daily_demand=round(avg_daily_demand, 2),
        lead_time_demand=lead_time_demand,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        current_on_hand_inventory=req.current_on_hand_inventory,
        reorder_recommended=reorder_recommended,
        spoilage_risk_alert=spoilage_alert,
        unit_cost_usd=unit_cost,
    )
