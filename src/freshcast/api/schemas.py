"""Pydantic schemas for FreshCast REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    model_loaded: bool = True
    active_dcs: int = 3
    active_skus: int = 10


class DailyForecast(BaseModel):
    date: str
    forecasted_demand: int
    lower_bound_p10: int
    upper_bound_p90: int
    unit_price: float
    is_promo: bool


class ForecastRequest(BaseModel):
    dc_id: str = Field(
        ...,
        description="Distribution Center ID",
        json_schema_extra={"example": "DC_BOS"},
    )
    sku_id: str = Field(
        ...,
        description="SKU Identifier",
        json_schema_extra={"example": "SKU_BEEF_SIRLOIN"},
    )
    horizon_days: int = Field(14, ge=1, le=30, description="Forecast horizon (1 to 30 days)")
    discount_pct: float = Field(
        0.0,
        ge=0.0,
        le=0.50,
        description="What-if promotional price discount (0.0 to 0.50)",
    )


class ForecastResponse(BaseModel):
    dc_id: str
    dc_name: str
    sku_id: str
    sku_name: str
    category: str
    horizon_days: int
    total_projected_demand: int
    average_daily_demand: float
    daily_forecasts: list[DailyForecast]


class ReplenishmentRequest(BaseModel):
    dc_id: str = Field(
        ...,
        description="Distribution Center ID",
        json_schema_extra={"example": "DC_BOS"},
    )
    sku_id: str = Field(
        ...,
        description="SKU Identifier",
        json_schema_extra={"example": "SKU_BEEF_SIRLOIN"},
    )
    service_level: float = Field(0.95, description="Cycle service level (0.90, 0.95, 0.98)")
    current_on_hand_inventory: int | None = Field(
        None, description="Current warehouse physical inventory (cases)"
    )


class ReplenishmentResponse(BaseModel):
    dc_id: str
    sku_id: str
    sku_name: str
    category: str
    shelf_life_days: int
    lead_time_days: int
    average_daily_demand: float
    lead_time_demand: float
    safety_stock: int
    reorder_point: int
    current_on_hand_inventory: int | None = None
    reorder_recommended: bool
    spoilage_risk_alert: bool
    unit_cost_usd: float
