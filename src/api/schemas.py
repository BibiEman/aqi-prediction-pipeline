"""
schemas.py

Pydantic schemas for the AQI 3-Day Forecast API.
"""

from typing import List

from pydantic import BaseModel, Field


# ============================================================
# Forecast Request
# ============================================================

class ForecastRequest(BaseModel):
    """
    Request schema for generating a 3-day AQI forecast
    for one city.
    """

    city: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="City name, for example Lahore",
    )


# ============================================================
# One Forecast Point
# ============================================================

class ForecastPoint(BaseModel):
    """
    One forecast point in the 3-day forecast.

    The model creates predictions every 3 hours.
    """

    timestamp: str

    forecast_step: int

    hours_ahead: int

    predicted_aqi: float

    aqi_category: str

    alert_level: str


# ============================================================
# Forecast Response
# ============================================================

class ForecastResponse(BaseModel):
    """
    Response schema for a complete 3-day AQI forecast.
    """

    city: str

    model: str

    version: str

    forecast_days: int

    forecast_interval_hours: int

    predictions_count: int

    forecast: List[ForecastPoint]

    status: str


# ============================================================
# Health Response
# ============================================================

class HealthResponse(BaseModel):
    """
    Optional health-check response schema.
    """

    status: str

    production_model: str

    production_version: str

    historical_rows: int

    cities: int