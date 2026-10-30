
"""API tests for FreshCast using the currently loaded dataset."""

import pytest
from fastapi.testclient import TestClient

from freshcast.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a test client and run the FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def metadata(client):
    """Fetch valid distribution-center and product IDs."""
    response = client.get("/api/v1/metadata")

    assert response.status_code == 200, response.text

    result = response.json()

    assert result.get("distribution_centers"), (
        "Metadata contains no distribution centers."
    )
    assert result.get("skus"), "Metadata contains no products."

    return result


@pytest.fixture(scope="module")
def valid_ids(metadata):
    """Select an actual distribution center and product from the dataset."""
    dc_id = metadata["distribution_centers"][0]["dc_id"]
    sku_id = metadata["skus"][0]["sku_id"]

    return {
        "dc_id": dc_id,
        "sku_id": sku_id,
    }


def test_health_endpoint(client):
    """The health endpoint should report a working API."""
    response = client.get("/health")

    assert response.status_code == 200, response.text

    data = response.json()

    assert data.get("status") == "healthy"
    assert data.get("model_loaded") is True
    assert data.get("active_dcs") == 3
    assert data.get("active_skus") == 10


def test_metadata_endpoint(client, metadata):
    """Metadata should list the real distribution centers and products."""
    assert len(metadata["distribution_centers"]) == 3
    assert len(metadata["skus"]) == 10

    for dc in metadata["distribution_centers"]:
        assert dc.get("dc_id")
        assert dc.get("dc_name")

    for sku in metadata["skus"]:
        assert sku.get("sku_id")
        assert sku.get("sku_name")


def test_forecast_endpoint(client, valid_ids):
    """Forecast should work for a valid real-data series."""
    payload = {
        **valid_ids,
        "horizon_days": 14,
        "discount_pct": 0.10,
    }

    response = client.post("/api/v1/forecast", json=payload)

    assert response.status_code == 200, (
        f"Forecast request failed: {response.status_code} {response.text}"
    )

    data = response.json()

    assert isinstance(data, dict)
    assert "total_projected_demand" in data
    assert "average_daily_demand" in data
    assert "daily_forecasts" in data

    assert isinstance(data["daily_forecasts"], list)
    assert len(data["daily_forecasts"]) == 14


def test_replenishment_endpoint(client, valid_ids):
    """Replenishment advice should work for a valid real-data series."""
    payload = {
        **valid_ids,
        "service_level": 0.95,
        "current_on_hand_inventory": 150,
    }

    response = client.post("/api/v1/replenish", json=payload)

    assert response.status_code == 200, (
        f"Replenishment request failed: "
        f"{response.status_code} {response.text}"
    )

    data = response.json()

    assert isinstance(data, dict)
    assert "safety_stock" in data
    assert "reorder_point" in data
    assert "reorder_recommended" in data
    assert "spoilage_risk_alert" in data
