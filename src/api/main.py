"""
main.py

FastAPI application for the AQI Prediction System.

Features
--------
- Health check
- Available cities
- Production model information
- Real-time AQI and weather monitoring
- 3-day AQI forecasting

Endpoints
---------
GET  /
GET  /health
GET  /model
GET  /cities
GET  /current/{city}
POST /forecast
"""

from pathlib import Path
from typing import Any, Dict

import json
import logging

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastPoint,
)

from src.model_training.config import (
    REGISTRY_FILE,
)

from src.model_training.utils import (
    load_dataset,
)

from src.model_training.predict import (
    FORECAST_DAYS,
    FORECAST_INTERVAL_HOURS,
    get_model_feature_columns,
    forecast_city,
    prepare_dataset,
)

from src.realtime.realtime_monitor import (
    get_current_conditions,
    SUPPORTED_CITIES,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="AQI Intelligence API",
    description=(
        "Production API for real-time AQI monitoring "
        "and 3-day air-quality forecasting."
    ),
    version="3.0.0",
)


# ============================================================
# Global Resources
# ============================================================

production_model = None

production_info: Dict[str, Any] = {}

historical_dataset = None

expected_model_features = None


# ============================================================
# Load Registry
# ============================================================

def load_registry() -> dict:
    """
    Load the local model registry.
    """

    registry_path = Path(
        REGISTRY_FILE
    )

    if not registry_path.exists():

        raise FileNotFoundError(
            "Model registry was not found:\n"
            f"{registry_path}"
        )

    with open(
        registry_path,
        "r",
        encoding="utf-8",
    ) as file:

        registry = json.load(
            file
        )

    return registry


# ============================================================
# Get Production Model Information
# ============================================================

def get_production_info():
    """
    Find the model/version currently marked
    as production in the registry.
    """

    registry = load_registry()

    models = registry.get(
        "models",
        {},
    )

    if not models:

        raise RuntimeError(
            "No models exist in the model registry."
        )

    for (
        model_name,
        model_data,
    ) in models.items():

        production_version = (
            model_data.get(
                "production_version"
            )
        )

        if not production_version:
            continue

        versions = model_data.get(
            "versions",
            [],
        )

        for version_info in versions:

            if (
                version_info.get(
                    "version"
                )
                == production_version
            ):

                return {
                    "model":
                        model_name,

                    "version":
                        production_version,

                    "metrics":
                        version_info.get(
                            "metrics",
                            {},
                        ),

                    "model_path":
                        version_info.get(
                            "model_path"
                        ),

                    "dataset_version":
                        version_info.get(
                            "dataset_version"
                        ),

                    "registered_at":
                        version_info.get(
                            "registered_at"
                        ),
                }

    raise RuntimeError(
        "No production model exists in the registry."
    )


# ============================================================
# Load Production Model
# ============================================================

def load_production_model():
    """
    Load the registered production model.
    """

    global production_model
    global production_info
    global expected_model_features

    print(
        "\n" + "=" * 70
    )
    print(
        "LOADING PRODUCTION MODEL"
    )
    print(
        "=" * 70
    )

    info = get_production_info()

    model_path = info.get(
        "model_path"
    )

    if not model_path:

        raise RuntimeError(
            "Production model path is missing."
        )

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Production model file "
            "does not exist:\n"
            f"{model_path}"
        )

    print(
        f"\nModel   : "
        f"{info['model']}"
    )

    print(
        f"Version : "
        f"{info['version']}"
    )

    print(
        f"Path    : "
        f"{model_path}"
    )

    production_model = joblib.load(
        model_path
    )

    production_info = info

    expected_model_features = (
        get_model_feature_columns(
            production_model
        )
    )

    print(
        "\nProduction model loaded successfully."
    )

    print(
        f"Expected features: "
        f"{len(expected_model_features)}"
    )


# ============================================================
# Load Historical Dataset
# ============================================================

def load_historical_data():
    """
    Load and prepare historical feature data
    used by recursive forecasting.
    """

    global historical_dataset

    print(
        "\n" + "=" * 70
    )
    print(
        "LOADING HISTORICAL DATA"
    )
    print(
        "=" * 70
    )

    df = load_dataset()

    df = prepare_dataset(
        df
    )

    historical_dataset = df

    print(
        f"\nRows   : "
        f"{len(df):,}"
    )

    print(
        f"Cities : "
        f"{df['city'].nunique()}"
    )

    print(
        f"Latest : "
        f"{df['timestamp'].max()}"
    )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    """
    API information.
    """

    return {
        "message":
            "AQI Intelligence API",

        "status":
            "running",

        "services": {
            "health":
                "/health",

            "model":
                "/model",

            "cities":
                "/cities",

            "current_conditions":
                "/current/{city}",

            "forecast":
                "/forecast",

            "documentation":
                "/docs",
        },
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    """
    Check whether required API resources are ready.
    """

    ready = (
        production_model is not None
        and historical_dataset is not None
        and bool(
            production_info
        )
    )

    if not ready:

        raise HTTPException(
            status_code=503,
            detail=(
                "AQI service is not ready."
            ),
        )

    return {
        "status":
            "healthy",

        "production_model":
            production_info[
                "model"
            ],

        "production_version":
            production_info[
                "version"
            ],

        "historical_rows":
            len(
                historical_dataset
            ),

        "cities":
            historical_dataset[
                "city"
            ].nunique(),

        "realtime_monitoring":
            True,

        "forecasting":
            True,
    }


# ============================================================
# Production Model Information
# ============================================================

@app.get("/model")
def model_info():
    """
    Return production-model metadata.
    """

    if not production_info:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model "
                "is not loaded."
            ),
        )

    return {
        "model":
            production_info[
                "model"
            ],

        "version":
            production_info[
                "version"
            ],

        "metrics":
            production_info.get(
                "metrics",
                {},
            ),

        "dataset_version":
            production_info.get(
                "dataset_version"
            ),

        "registered_at":
            production_info.get(
                "registered_at"
            ),

        "forecast_days":
            FORECAST_DAYS,

        "forecast_interval_hours":
            FORECAST_INTERVAL_HOURS,
    }


# ============================================================
# Available Cities
# ============================================================

@app.get("/cities")
def cities():
    """
    Return cities available for monitoring/forecasting.
    """

    if historical_dataset is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Historical dataset "
                "is not loaded."
            ),
        )

    historical_cities = sorted(
        historical_dataset[
            "city"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Only expose cities supported by both
    # the live monitor and historical forecasting.
    supported_lookup = {
        city.lower(): city
        for city in SUPPORTED_CITIES
    }

    common_cities = []

    for city in historical_cities:

        canonical = supported_lookup.get(
            city.lower()
        )

        if canonical:
            common_cities.append(
                canonical
            )

    return {
        "count":
            len(common_cities),

        "cities":
            common_cities,
    }


# ============================================================
# Real-Time Current Conditions
# ============================================================

@app.get("/current/{city}")
def current_conditions(
    city: str,
):
    """
    Return real-time weather, pollutant,
    AQI category and health guidance for a city.
    """

    city_lookup = {
        name.lower(): name
        for name in SUPPORTED_CITIES
    }

    requested_city = (
        city
        .strip()
        .lower()
    )

    canonical_city = (
        city_lookup.get(
            requested_city
        )
    )

    if canonical_city is None:

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"City '{city}' "
                    "is not supported."
                ),
                "available_cities":
                    SUPPORTED_CITIES,
            },
        )

    try:

        logger.info(
            "Fetching real-time conditions for %s",
            canonical_city,
        )

        result = get_current_conditions(
            canonical_city
        )

        return {
            "status":
                "success",

            **result,
        }

    except Exception as error:

        logger.exception(
            "Real-time monitoring failed for %s",
            canonical_city,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve current "
                f"conditions for {canonical_city}. "
                f"Error: {error}"
            ),
        )


# ============================================================
# 3-Day Forecast
# ============================================================

@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
):
    """
    Generate a 72-hour AQI forecast for one city.
    """

    if production_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model "
                "is not loaded."
            ),
        )

    if historical_dataset is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Historical dataset "
                "is not loaded."
            ),
        )

    try:

        # ----------------------------------------------------
        # Match city
        # ----------------------------------------------------

        city_matches = (
            historical_dataset[
                historical_dataset[
                    "city"
                ]
                .astype(str)
                .str.lower()
                == request.city.lower()
            ]
            .copy()
        )

        if city_matches.empty:

            available = sorted(
                historical_dataset[
                    "city"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            raise HTTPException(
                status_code=404,
                detail={
                    "message": (
                        f"City "
                        f"'{request.city}' "
                        "was not found."
                    ),
                    "available_cities":
                        available,
                },
            )

        city_name = str(
            city_matches[
                "city"
            ].iloc[0]
        )

        logger.info(
            "Generating 3-day forecast for %s",
            city_name,
        )

        # ----------------------------------------------------
        # Generate Forecast
        # ----------------------------------------------------

        forecast_df = forecast_city(
            model=production_model,
            city_data=city_matches,
            city=city_name,
            expected_columns=(
                expected_model_features
            ),
        )

        if forecast_df.empty:

            raise RuntimeError(
                "Forecast returned no rows."
            )

        # ----------------------------------------------------
        # Convert to response objects
        # ----------------------------------------------------

        forecast_points = []

        for _, row in forecast_df.iterrows():

            forecast_points.append(
                ForecastPoint(
                    timestamp=str(
                        row[
                            "timestamp"
                        ]
                    ),

                    forecast_step=int(
                        row[
                            "forecast_step"
                        ]
                    ),

                    hours_ahead=int(
                        row[
                            "hours_ahead"
                        ]
                    ),

                    predicted_aqi=float(
                        row[
                            "predicted_aqi"
                        ]
                    ),

                    aqi_category=str(
                        row[
                            "aqi_category"
                        ]
                    ),

                    alert_level=str(
                        row[
                            "alert_level"
                        ]
                    ),
                )
            )

        return ForecastResponse(
            city=city_name,

            model=production_info[
                "model"
            ],

            version=production_info[
                "version"
            ],

            forecast_days=(
                FORECAST_DAYS
            ),

            forecast_interval_hours=(
                FORECAST_INTERVAL_HOURS
            ),

            predictions_count=(
                len(
                    forecast_points
                )
            ),

            forecast=forecast_points,

            status="success",
        )

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Forecast failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast generation failed. "
                f"Error: {error}"
            ),
        )


# ============================================================
# Startup
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():
    """
    Load production resources once when
    the FastAPI application starts.
    """

    try:

        load_production_model()

        load_historical_data()

        print(
            "\n" + "=" * 70
        )
        print(
            "AQI API READY"
        )
        print(
            "=" * 70
        )

        print(
            "\nProduction Model : "
            f"{production_info['model']}"
        )

        print(
            "Version          : "
            f"{production_info['version']}"
        )

        print(
            "Forecast Horizon : "
            f"{FORECAST_DAYS} days"
        )

        print(
            "Forecast Interval: "
            f"{FORECAST_INTERVAL_HOURS} hours"
        )

        print(
            "Real-Time Monitor: enabled"
        )

        print(
            "Supported Cities : "
            f"{len(SUPPORTED_CITIES)}"
        )

    except Exception as error:

        logger.exception(
            "API startup failed."
        )

        print(
            "\nAQI API startup failed:"
        )

        print(
            error
        )