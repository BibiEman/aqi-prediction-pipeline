"""
Integration tests for the AQI Production Forecasting API.

These tests validate the current production API contract:
- service health
- production model metadata
- supported cities
- current city conditions
- GET forecast
- POST forecast
- invalid city handling
"""

from fastapi.testclient import TestClient

from src.api.main import app


# ============================================================
# Helpers
# ============================================================

def assert_forecast_response(data, city):
    """Validate the common 3-day forecast response structure."""

    assert data["status"] == "success"

    assert data["city"] == city

    assert data["model"] == "LightGBM"
    assert data["version"] == "v1"

    assert data["forecast_days"] == 3
    assert data["forecast_interval_hours"] == 3

    assert data["predictions_count"] == 24

    assert data["data_source"] in {
        "REALTIME",
        "HISTORICAL_FALLBACK",
    }

    assert isinstance(data["forecast"], list)
    assert len(data["forecast"]) == 24

    first = data["forecast"][0]

    required_fields = {
        "timestamp",
        "forecast_step",
        "hours_ahead",
        "predicted_aqi",
        "aqi_category",
        "alert_level",
        "health_guidance",
        "data_source",
        "forecast_method",
    }

    assert required_fields.issubset(first.keys())

    assert first["forecast_step"] == 1
    assert first["hours_ahead"] == 3

    assert isinstance(
        first["predicted_aqi"],
        (int, float),
    )

    assert first["data_source"] in {
        "REALTIME",
        "HISTORICAL_FALLBACK",
    }


# ============================================================
# API Tests
# ============================================================

def test_root():
    """Root endpoint should report a running API."""

    with TestClient(app) as client:

        response = client.get("/")

        assert response.status_code == 200

        data = response.json()

        assert isinstance(data, dict)

        assert data.get("status") in {
            "running",
            "healthy",
            "ok",
        }


def test_health():
    """Health endpoint should report production readiness."""

    with TestClient(app) as client:

        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"

        assert data["production_model"] == "LightGBM"
        assert data["production_version"] == "v1"

        assert data["realtime_rows"] > 0
        assert data["historical_rows"] > 0

        assert data["cities"] == 10

        assert data["forecasting"] is True
        assert data["realtime_monitoring"] is True


def test_model():
    """Model endpoint should expose production model metadata."""

    with TestClient(app) as client:

        response = client.get("/model")

        assert response.status_code == 200

        data = response.json()

        assert data["model"] == "LightGBM"
        assert data["version"] == "v1"

        assert data["forecast_days"] == 3
        assert data["forecast_interval_hours"] == 3
        assert data["forecast_points_per_city"] == 24

        metrics = data["metrics"]

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

        assert metrics["RMSE"] >= 0


def test_cities():
    """Cities endpoint should expose all supported cities."""

    with TestClient(app) as client:

        response = client.get("/cities")

        assert response.status_code == 200

        data = response.json()

        assert data["count"] == 10

        assert isinstance(
            data["cities"],
            list,
        )

        assert len(data["cities"]) == 10

        assert "Lahore" in data["cities"]
        assert "Karachi" in data["cities"]
        assert "Islamabad" in data["cities"]


def test_current_lahore():
    """Current conditions should be available for Lahore."""

    with TestClient(app) as client:

        response = client.get(
            "/current/Lahore"
        )

        assert response.status_code == 200

        data = response.json()

        assert isinstance(data, dict)

        assert data.get("city") == "Lahore"


def test_get_forecast_lahore():
    """GET forecast should return 24 production predictions."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/Lahore"
        )

        assert response.status_code == 200

        data = response.json()

        assert_forecast_response(
            data,
            "Lahore",
        )


def test_post_forecast_lahore():
    """POST forecast should support the ForecastRequest schema."""

    with TestClient(app) as client:

        response = client.post(
            "/forecast",
            json={
                "city": "Lahore",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert_forecast_response(
            data,
            "Lahore",
        )


def test_forecast_case_insensitive():
    """City lookup should ideally tolerate lowercase input."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/lahore"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["city"] == "Lahore"

        assert data["predictions_count"] == 24


def test_invalid_city():
    """Unsupported cities should not generate a forecast."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/InvalidCityXYZ"
        )

        assert response.status_code in {
            400,
            404,
            422,
        }


def test_forecast_points():
    """Forecast points should progress from 3 to 72 hours."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/Lahore"
        )

        assert response.status_code == 200

        data = response.json()

        forecast = data["forecast"]

        assert len(forecast) == 24

        assert forecast[0]["forecast_step"] == 1
        assert forecast[0]["hours_ahead"] == 3

        assert forecast[-1]["forecast_step"] == 24
        assert forecast[-1]["hours_ahead"] == 72


def test_forecast_source_consistency():
    """Top-level and point-level data sources should agree."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/Lahore"
        )

        assert response.status_code == 200

        data = response.json()

        source = data["data_source"]

        for point in data["forecast"]:

            assert point["data_source"] == source


def test_forecast_aqi_values():
    """Every forecast should contain a valid numeric AQI."""

    with TestClient(app) as client:

        response = client.get(
            "/forecast/Lahore"
        )

        assert response.status_code == 200

        forecast = response.json()["forecast"]

        for point in forecast:

            assert isinstance(
                point["predicted_aqi"],
                (int, float),
            )

            assert point["predicted_aqi"] >= 0