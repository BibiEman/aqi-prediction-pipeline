"""
schemas.py

Pydantic response/request schemas for the
AQI Production Forecasting API.
"""

from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# Forecast Request
# ============================================================

class ForecastRequest(BaseModel):
    """
    Request one 3-day AQI forecast.
    """

    city: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Supported city, e.g. Lahore",
        examples=["Lahore"],
    )


# ============================================================
# Forecast Point
# ============================================================

class ForecastPoint(BaseModel):
    """
    One +3-hour forecast point.
    """

    timestamp: str

    forecast_step: int = Field(
        ge=1
    )

    hours_ahead: int = Field(
        ge=3,
        le=72,
    )

    predicted_aqi: float = Field(
        ge=0,
        le=500,
    )

    aqi_category: str

    alert_level: str

    health_guidance: Optional[str] = None

    data_source: Optional[str] = None

    forecast_method: Optional[str] = None


# ============================================================
# Forecast Response
# ============================================================

class ForecastResponse(BaseModel):
    """
    Complete 3-day city forecast.
    """

    city: str

    model: str

    version: str

    forecast_days: int

    forecast_interval_hours: int

    predictions_count: int

    data_source: str

    forecast: List[
        ForecastPoint
    ]

    status: str


# ============================================================
# Health Response
# ============================================================

class HealthResponse(BaseModel):
    """
    API health information.
    """

    status: str

    production_model: str

    production_version: str

    realtime_rows: int

    historical_rows: int

    cities: int

    forecasting: bool

    realtime_monitoring: bool