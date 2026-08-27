"""
main.py

Production FastAPI application for the
AQI Prediction System.

Capabilities
------------
- API health checking
- production model information
- supported cities
- current real-time AQI/weather
- recursive 3-day AQI forecasting
- production registry integration
- real-time-first forecasting
- historical fallback while live history accumulates

Endpoints
---------
GET  /
GET  /health
GET  /model
GET  /cities
GET  /current/{city}
POST /forecast
GET  /forecast/{city}
"""

import logging
from typing import (
    Any,
    Dict,
    Optional,
)

import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)


from src.api.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)


from src.model_training.predict import (
    FORECAST_DAYS,
    FORECAST_INTERVAL_HOURS,
    SUPPORTED_CITIES,
    generate_3day_forecast,
    get_model_feature_columns,
    load_historical_dataset,
    load_production_model,
    load_realtime_dataset,
    select_city_forecast_source,
    forecast_city,
)


from src.realtime.realtime_monitor import (
    get_current_conditions,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=(
        "AQI Intelligence API"
    ),
    description=(
        "Production AQI monitoring and "
        "3-day forecasting API."
    ),
    version="4.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=False,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# ============================================================
# Global Production Resources
# ============================================================

production_model = None

production_info: Dict[
    str,
    Any,
] = {}

expected_model_features = None

historical_dataset: Optional[
    pd.DataFrame
] = None

realtime_dataset: Optional[
    pd.DataFrame
] = None


# ============================================================
# City Normalization
# ============================================================

def get_canonical_city(
    city: str,
) -> str:
    """
    Validate and normalize a supported city.
    """

    lookup = {
        supported.lower():
            supported

        for supported
        in SUPPORTED_CITIES
    }

    requested = (
        city
        .strip()
        .lower()
    )

    canonical = (
        lookup.get(
            requested
        )
    )

    if canonical is None:

        raise HTTPException(
            status_code=404,
            detail={
                "message":
                    (
                        f"City '{city}' "
                        "is not supported."
                    ),

                "available_cities":
                    SUPPORTED_CITIES,
            },
        )

    return canonical


# ============================================================
# Load Production Resources
# ============================================================

def initialize_production_resources():
    """
    Load registry production model and datasets.
    """

    global production_model
    global production_info
    global expected_model_features
    global historical_dataset
    global realtime_dataset

    print(
        "\n" + "=" * 70
    )

    print(
        "INITIALIZING AQI PRODUCTION API"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Production model
    # --------------------------------------------------------

    (
        production_model,
        production_info,
    ) = load_production_model()

    expected_model_features = (
        get_model_feature_columns(
            production_model
        )
    )

    # --------------------------------------------------------
    # Historical fallback
    # --------------------------------------------------------

    try:

        historical_dataset = (
            load_historical_dataset()
        )

    except Exception as error:

        logger.warning(
            "Historical dataset "
            "could not be loaded: %s",
            error,
        )

        historical_dataset = None

    # --------------------------------------------------------
    # Real-time features
    # --------------------------------------------------------

    try:

        realtime_dataset = (
            load_realtime_dataset()
        )

    except Exception as error:

        logger.warning(
            "Real-time dataset "
            "could not be loaded: %s",
            error,
        )

        realtime_dataset = None

    if (
        historical_dataset is None
        and
        realtime_dataset is None
    ):

        raise RuntimeError(
            "No forecasting dataset is available."
        )

    print(
        "\nProduction resources loaded."
    )

    print(
        f"Model   : "
        f"{production_info['model_name']}"
    )

    print(
        f"Version : "
        f"{production_info['version']}"
    )

    print(
        f"Features: "
        f"{len(expected_model_features)}"
    )

    if realtime_dataset is not None:

        print(
            f"Real-time rows : "
            f"{len(realtime_dataset):,}"
        )

    if historical_dataset is not None:

        print(
            f"Historical rows: "
            f"{len(historical_dataset):,}"
        )


# ============================================================
# Refresh Real-Time Dataset
# ============================================================

def refresh_realtime_dataset():
    """
    Reload real-time feature file before forecasting.

    This prevents the API from using a stale copy loaded
    only when the server started.
    """

    global realtime_dataset

    try:

        latest = (
            load_realtime_dataset()
        )

        if latest is not None:

            realtime_dataset = latest

    except Exception as error:

        logger.warning(
            "Could not refresh real-time "
            "features: %s",
            error,
        )


# ============================================================
# Build Forecast Response
# ============================================================

def create_forecast_response(
    city: str,
) -> ForecastResponse:
    """
    Generate one city's production forecast.
    """

    if production_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model "
                "is unavailable."
            ),
        )

    canonical_city = (
        get_canonical_city(
            city
        )
    )

    refresh_realtime_dataset()

    try:

        (
            city_data,
            source_name,
        ) = (
            select_city_forecast_source(
                city=canonical_city,

                realtime_df=(
                    realtime_dataset
                ),

                historical_df=(
                    historical_dataset
                ),

                expected_columns=(
                    expected_model_features
                ),
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=(
                "No suitable forecasting "
                f"data is available for "
                f"{canonical_city}. "
                f"Error: {error}"
            ),
        ) from error

    logger.info(
        "Generating 3-day forecast "
        "for %s using %s",
        canonical_city,
        source_name,
    )

    try:

        forecast_df = (
            forecast_city(
                model=production_model,

                city_data=city_data,

                city=canonical_city,

                expected_columns=(
                    expected_model_features
                ),

                production_info=(
                    production_info
                ),

                source_name=(
                    source_name
                ),
            )
        )

    except Exception as error:

        logger.exception(
            "Forecast generation failed "
            "for %s",
            canonical_city,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast generation failed "
                f"for {canonical_city}. "
                f"Error: {error}"
            ),
        ) from error

    if forecast_df.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast pipeline returned "
                "no predictions."
            ),
        )

    points = []

    for _, row in (
        forecast_df.iterrows()
    ):

        points.append(
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

                health_guidance=str(
                    row.get(
                        "health_guidance",
                        "",
                    )
                ),

                data_source=str(
                    row.get(
                        "data_source",
                        source_name,
                    )
                ),

                forecast_method=str(
                    row.get(
                        "forecast_method",
                        (
                            "recursive_"
                            "persistence"
                        ),
                    )
                ),
            )
        )

    return ForecastResponse(
        city=canonical_city,

        model=production_info[
            "model_name"
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
            len(points)
        ),

        data_source=(
            source_name
        ),

        forecast=points,

        status="success",
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
        "service":
            "AQI Intelligence API",

        "version":
            "4.0.0",

        "status":
            "running",

        "endpoints": {
            "health":
                "/health",

            "model":
                "/model",

            "cities":
                "/cities",

            "current":
                "/current/{city}",

            "forecast_post":
                "/forecast",

            "forecast_get":
                "/forecast/{city}",

            "documentation":
                "/docs",
        },
    }


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Return API readiness information.
    """

    if production_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model "
                "is not loaded."
            ),
        )

    realtime_rows = (
        len(
            realtime_dataset
        )
        if realtime_dataset
        is not None
        else 0
    )

    historical_rows = (
        len(
            historical_dataset
        )
        if historical_dataset
        is not None
        else 0
    )

    available_cities = set()

    if realtime_dataset is not None:

        available_cities.update(
            realtime_dataset[
                "city"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    if historical_dataset is not None:

        available_cities.update(
            historical_dataset[
                "city"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    return HealthResponse(
        status="healthy",

        production_model=(
            production_info[
                "model_name"
            ]
        ),

        production_version=(
            production_info[
                "version"
            ]
        ),

        realtime_rows=(
            realtime_rows
        ),

        historical_rows=(
            historical_rows
        ),

        cities=(
            len(
                available_cities
            )
        ),

        forecasting=True,

        realtime_monitoring=True,
    )


# ============================================================
# Model Information
# ============================================================

@app.get(
    "/model"
)
def model_information():
    """
    Return production-model metadata.
    """

    if not production_info:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production registry "
                "is unavailable."
            ),
        )

    return {
        "model":
            production_info[
                "model_name"
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

        "model_path":
            production_info.get(
                "resolved_model_path"
            ),

        "forecast_days":
            FORECAST_DAYS,

        "forecast_interval_hours":
            FORECAST_INTERVAL_HOURS,

        "forecast_points_per_city":
            (
                FORECAST_DAYS
                * 24
                // FORECAST_INTERVAL_HOURS
            ),
    }


# ============================================================
# Available Cities
# ============================================================

@app.get(
    "/cities"
)
def cities():
    """
    Return supported cities.
    """

    return {
        "count":
            len(
                SUPPORTED_CITIES
            ),

        "cities":
            SUPPORTED_CITIES,
    }


# ============================================================
# Current Conditions
# ============================================================

@app.get(
    "/current/{city}"
)
def current_conditions(
    city: str,
):
    """
    Return current weather, pollutants and AQI.
    """

    canonical_city = (
        get_canonical_city(
            city
        )
    )

    try:

        logger.info(
            "Fetching current conditions "
            "for %s",
            canonical_city,
        )

        result = (
            get_current_conditions(
                canonical_city
            )
        )

        return {
            "status":
                "success",

            **result,
        }

    except Exception as error:

        logger.exception(
            "Current-condition request "
            "failed for %s",
            canonical_city,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve current "
                f"conditions for "
                f"{canonical_city}. "
                f"Error: {error}"
            ),
        ) from error


# ============================================================
# Forecast - POST
# ============================================================

@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast_post(
    request: ForecastRequest,
):
    """
    Generate one 3-day AQI forecast.
    """

    return (
        create_forecast_response(
            request.city
        )
    )


# ============================================================
# Forecast - GET
# ============================================================

@app.get(
    "/forecast/{city}",
    response_model=ForecastResponse,
)
def forecast_get(
    city: str,
):
    """
    Convenience GET endpoint for dashboard usage.
    """

    return (
        create_forecast_response(
            city
        )
    )


# ============================================================
# Refresh Production Resources
# ============================================================

@app.post(
    "/refresh"
)
def refresh_resources():
    """
    Reload the production registry/model and latest datasets.

    Useful after a new model version is promoted.
    """

    try:

        initialize_production_resources()

        return {
            "status":
                "success",

            "message":
                "Production resources refreshed.",

            "model":
                production_info[
                    "model_name"
                ],

            "version":
                production_info[
                    "version"
                ],
        }

    except Exception as error:

        logger.exception(
            "Resource refresh failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not refresh "
                "production resources. "
                f"Error: {error}"
            ),
        ) from error


# ============================================================
# Startup
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():
    """
    Load production resources when API starts.
    """

    try:

        initialize_production_resources()

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
            f"\nProduction Model : "
            f"{production_info['model_name']}"
        )

        print(
            f"Version          : "
            f"{production_info['version']}"
        )

        print(
            f"Forecast Horizon : "
            f"{FORECAST_DAYS} days"
        )

        print(
            f"Forecast Interval: "
            f"{FORECAST_INTERVAL_HOURS} hours"
        )

        print(
            f"Supported Cities : "
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