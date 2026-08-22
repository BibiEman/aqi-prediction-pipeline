
"""
tests/test_api.py

Automated tests for the AQI Prediction API.
"""

from fastapi.testclient import TestClient

from src.api.main import app, load_production_model


# ============================================================
# Load production model before tests
# ============================================================

try:
    load_production_model()
except Exception as error:
    raise RuntimeError(
        f"Could not load production model for API tests: {error}"
    )


client = TestClient(app)


# ============================================================
# Root Endpoint
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["health"] == "/health"
    assert data["model"] == "/model"
    assert data["prediction"] == "/predict"


# ============================================================
# Health Endpoint
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["production_model"] == "LightGBM"
    assert data["production_version"] == "v1"


# ============================================================
# Model Endpoint
# ============================================================

def test_model():

    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "LightGBM"
    assert data["version"] == "v1"

    assert "metrics" in data

    assert "MAE" in data["metrics"]
    assert "RMSE" in data["metrics"]
    assert "R2" in data["metrics"]
    assert "MAPE" in data["metrics"]


# ============================================================
# Prediction Endpoint
# ============================================================

def test_prediction():

    payload = {
        "city": "Lahore",

        "latitude": 31.5204,
        "longitude": 74.3587,

        "pm2_5": 100,
        "pm10": 150,

        "carbon_monoxide": 500,
        "nitrogen_dioxide": 40,
        "sulphur_dioxide": 10,
        "ozone": 80,

        "temperature": 30,
        "humidity": 60,
        "pressure": 1005,

        "wind_speed": 10,
        "wind_direction": 180,

        "precipitation": 0,
        "cloud_cover": 20,

        "us_aqi": 120,

        "aqi_lag_1": 130,
        "aqi_lag_3": 125,
        "aqi_lag_6": 120,
        "aqi_lag_24": 115,

        "aqi_roll_3": 128,
        "aqi_roll_6": 125,
        "aqi_roll_24": 120,

        "hour": 20,
        "day": 30,
        "month": 6,
        "day_of_week": 0,

        "season": "summer",
        "year": 2025,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["city"] == "Lahore"
    assert data["model"] == "LightGBM"
    assert data["version"] == "v1"

    assert data["forecast_horizon_hours"] == 3

    assert isinstance(
        data["predicted_aqi"],
        (int, float),
    )

    assert data["predicted_aqi"] >= 0

    assert data["status"] == "success"


# ============================================================
# Invalid Input Test
# ============================================================

def test_invalid_prediction_request():

    payload = {
        "city": "Lahore",

        "latitude": 31.5204,
        "longitude": 74.3587,

        "pm2_5": 100,
        "pm10": 150,

        "carbon_monoxide": 500,
        "nitrogen_dioxide": 40,
        "sulphur_dioxide": 10,
        "ozone": 80,

        "temperature": 30,
        "humidity": 60,
        "pressure": 1005,

        "wind_speed": 10,
        "wind_direction": 180,

        "precipitation": 0,
        "cloud_cover": 20,

        "us_aqi": 120,

        "aqi_lag_1": 130,
        "aqi_lag_3": 125,
        "aqi_lag_6": 120,
        "aqi_lag_24": 115,

        "aqi_roll_3": 128,
        "aqi_roll_6": 125,
        "aqi_roll_24": 120,

        # Invalid value: allowed range is 0-23
        "hour": 25,

        "day": 30,
        "month": 6,
        "day_of_week": 0,

        "season": "summer",
        "year": 2025,
    }

    response = client.post(
        "/predict",
        json=payload,
    )
    assert response.status_code == 422
