"""
predict.py

Production 3-Day AQI Forecasting Pipeline.

Purpose
-------
Generate AQI forecasts for the next 72 hours using the
currently promoted production model from the local model registry.

Forecast configuration
----------------------
Model target:
    target_aqi(t) = us_aqi(t + 3 hours)

Forecast horizon:
    72 hours = 3 days

Forecast interval:
    3 hours

Forecast points:
    24 predictions per city

Production architecture
-----------------------
Latest real-time features
        ↓
Production model registry
        ↓
Production sklearn pipeline
        ↓
Recursive +3-hour forecasting
        ↓
3-day AQI forecast

Important
---------
The serialized production model already contains:

    ColumnTransformer
        ↓
    OneHotEncoder
        ↓
    LightGBM

Therefore categorical variables are NOT encoded manually.

Future weather and pollutant variables currently use a
persistence assumption. AQI autoregressive features are updated
recursively.

This module is cross-platform and works on:

    Windows
    Linux
    GitHub Actions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

REGISTRY_FILE = (
    MODELS_DIR
    / "registry.json"
)

HISTORICAL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_training_dataset.csv"
)

REALTIME_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / "realtime_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

PREDICTION_DIR = (
    RESULTS_DIR
    / "predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FORECAST_FILE = (
    PREDICTION_DIR
    / "aqi_3day_forecast.csv"
)


# ============================================================
# Forecast Configuration
# ============================================================

FORECAST_DAYS = 3

FORECAST_INTERVAL_HOURS = 3

FORECAST_HOURS = (
    FORECAST_DAYS
    * 24
)

FORECAST_STEPS = (
    FORECAST_HOURS
    // FORECAST_INTERVAL_HOURS
)

MINIMUM_HISTORY_HOURS = 24


# ============================================================
# Model Columns
# ============================================================

NON_MODEL_COLUMNS = [
    "timestamp",
    "target_aqi",
    "model_ready",
]


# ============================================================
# Supported Cities
# ============================================================

SUPPORTED_CITIES = [
    "Faisalabad",
    "Hyderabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rawalpindi",
    "Sialkot",
]


# ============================================================
# City Normalization
# ============================================================

CITY_NORMALIZATION = {
    "Siālkot": "Sialkot",
    "Sialkot": "Sialkot",
}


def normalize_city_name(
    city,
):
    """
    Normalize inconsistent API city spellings.
    """

    city = str(
        city
    ).strip()

    return CITY_NORMALIZATION.get(
        city,
        city,
    )


# ============================================================
# AQI Classification
# ============================================================

def get_aqi_category(
    aqi,
):
    """
    Return US AQI category.
    """

    if aqi is None:
        return "Unknown"

    aqi = float(
        aqi
    )

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups"
        )

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def get_alert_level(
    aqi,
):
    """
    Convert AQI into operational alert level.
    """

    if aqi is None:
        return "Unknown"

    aqi = float(
        aqi
    )

    if aqi <= 100:
        return "Normal"

    if aqi <= 150:
        return "Caution"

    if aqi <= 200:
        return "Warning"

    if aqi <= 300:
        return "High Alert"

    return "Emergency"


def get_health_guidance(
    aqi,
):
    """
    Return basic public-health guidance.
    """

    if aqi is None:
        return (
            "AQI information is unavailable."
        )

    if aqi <= 50:

        return (
            "Air quality is satisfactory. "
            "Normal outdoor activity may continue."
        )

    if aqi <= 100:

        return (
            "Air quality is acceptable for most people. "
            "Very sensitive individuals should monitor exposure."
        )

    if aqi <= 150:

        return (
            "Sensitive groups should reduce prolonged "
            "or strenuous outdoor activity."
        )

    if aqi <= 200:

        return (
            "Everyone may experience health effects. "
            "Sensitive groups should avoid prolonged "
            "outdoor exposure."
        )

    if aqi <= 300:

        return (
            "Health alert conditions are expected. "
            "Significantly reduce outdoor activity."
        )

    return (
        "Hazardous air quality. Avoid unnecessary outdoor "
        "activity and follow public-health guidance."
    )


# ============================================================
# Season
# ============================================================

def get_season(
    month,
):
    """
    Convert month to season.
    """

    month = int(
        month
    )

    if month in [
        12,
        1,
        2,
    ]:
        return "Winter"

    if month in [
        3,
        4,
        5,
    ]:
        return "Spring"

    if month in [
        6,
        7,
        8,
    ]:
        return "Summer"

    return "Autumn"


# ============================================================
# Registry
# ============================================================

def load_registry():
    """
    Load local model registry metadata.
    """

    print("\n" + "=" * 70)
    print("LOADING MODEL REGISTRY")
    print("=" * 70)

    if not REGISTRY_FILE.exists():

        raise FileNotFoundError(
            "Model registry was not found.\n"
            f"Expected:\n{REGISTRY_FILE}"
        )

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(
                file
            )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Model registry contains invalid JSON.\n"
            f"File:\n{REGISTRY_FILE}"
        ) from error

    print(
        f"\nRegistry:\n{REGISTRY_FILE}"
    )

    return registry


# ============================================================
# Production Model Information
# ============================================================

def get_production_model_info(
    registry,
):
    """
    Find the currently promoted production model.
    """

    models = registry.get(
        "models",
        {},
    )

    if not models:

        raise RuntimeError(
            "Model registry contains no models."
        )

    for (
        model_name,
        model_info,
    ) in models.items():

        production_version = (
            model_info.get(
                "production_version"
            )
        )

        if not production_version:
            continue

        versions = model_info.get(
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
                    "model_name":
                        model_name,

                    "version":
                        production_version,

                    "metrics":
                        version_info.get(
                            "metrics",
                            {},
                        ),

                    "dataset_version":
                        version_info.get(
                            "dataset_version"
                        ),

                    "registered_at":
                        version_info.get(
                            "registered_at"
                        ),

                    "status":
                        version_info.get(
                            "status",
                            "production",
                        ),
                }

    raise RuntimeError(
        "No production model is currently "
        "defined in registry.json."
    )


# ============================================================
# Portable Registry Model Path
# ============================================================

def resolve_production_model_path(
    production_info,
):
    """
    Resolve production model relative to PROJECT_ROOT.

    Never trusts machine-specific absolute paths stored in
    historical registry metadata.
    """

    model_name = (
        production_info[
            "model_name"
        ]
    )

    version = str(
        production_info[
            "version"
        ]
    )

    if not version.startswith(
        "v"
    ):

        version = (
            f"v{version}"
        )

    return (
        MODELS_DIR
        / "registry"
        / model_name
        / version
        / "model.pkl"
    )


# ============================================================
# Load Production Model
# ============================================================

def load_production_model():
    """
    Load currently promoted production sklearn pipeline.

    Returns
    -------
    model
        Complete sklearn preprocessing/model pipeline.

    production_info : dict
        Production registry metadata.
    """

    registry = load_registry()

    production_info = (
        get_production_model_info(
            registry
        )
    )

    model_path = (
        resolve_production_model_path(
            production_info
        )
    )

    print("\n" + "=" * 70)
    print("LOADING PRODUCTION AQI MODEL")
    print("=" * 70)

    print(
        f"\nModel   : "
        f"{production_info['model_name']}"
    )

    print(
        f"Version : "
        f"{production_info['version']}"
    )

    print(
        f"Path    : "
        f"{model_path}"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Production model artifact "
            "was not found.\n\n"
            f"Expected:\n{model_path}"
        )

    try:

        model = joblib.load(
            model_path
        )

    except Exception as error:

        raise RuntimeError(
            "Production model could not be loaded.\n"
            f"Path: {model_path}\n"
            f"Error: {error}"
        ) from error

    print(
        "\nProduction model loaded successfully."
    )

    production_info[
        "resolved_model_path"
    ] = str(
        model_path
    )

    return (
        model,
        production_info,
    )


# ============================================================
# Model Feature Names
# ============================================================

def get_model_feature_columns(
    model,
):
    """
    Get raw feature names expected by sklearn pipeline.
    """

    if not hasattr(
        model,
        "named_steps",
    ):

        raise RuntimeError(
            "Production artifact is not "
            "an sklearn Pipeline."
        )

    if (
        "preprocessor"
        not in model.named_steps
    ):

        raise RuntimeError(
            "Production pipeline does not contain "
            "the expected 'preprocessor' step."
        )

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    if hasattr(
        preprocessor,
        "feature_names_in_",
    ):

        return list(
            preprocessor.feature_names_in_
        )

    raise RuntimeError(
        "Could not determine production "
        "model input feature names."
    )


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_dataset(
    dataframe,
):
    """
    Clean and normalize a forecasting source dataset.
    """

    if dataframe is None:

        raise ValueError(
            "Forecast dataset is None."
        )

    if dataframe.empty:

        raise ValueError(
            "Forecast dataset is empty."
        )

    dataframe = dataframe.copy()

    required = [
        "timestamp",
        "city",
        "us_aqi",
    ]

    missing = [
        column
        for column in required
        if column
        not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Forecast dataset is missing "
            "required columns:\n"
            f"{missing}"
        )

    dataframe[
        "timestamp"
    ] = pd.to_datetime(
        dataframe[
            "timestamp"
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "timestamp"
        ].notna()
    ].copy()

    dataframe[
        "city"
    ] = (
        dataframe[
            "city"
        ]
        .apply(
            normalize_city_name
        )
    )

    dataframe[
        "us_aqi"
    ] = pd.to_numeric(
        dataframe[
            "us_aqi"
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "us_aqi"
        ].notna()
    ].copy()

    dataframe = (
        dataframe
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .drop_duplicates(
            subset=[
                "city",
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# Historical Dataset
# ============================================================

def load_historical_dataset():
    """
    Load historical model-training dataset.
    """

    if not HISTORICAL_DATA_FILE.exists():

        return None

    print(
        "\nLoading historical forecasting data..."
    )

    dataframe = pd.read_csv(
        HISTORICAL_DATA_FILE
    )

    dataframe = prepare_dataset(
        dataframe
    )

    print(
        f"Historical rows : "
        f"{len(dataframe):,}"
    )

    print(
        f"Historical end  : "
        f"{dataframe['timestamp'].max()}"
    )

    return dataframe


# ============================================================
# Real-Time Dataset
# ============================================================

def load_realtime_dataset():
    """
    Load locally persisted real-time engineered features.
    """

    if not REALTIME_FEATURE_FILE.exists():

        return None

    print(
        "\nLoading real-time forecasting features..."
    )

    dataframe = pd.read_csv(
        REALTIME_FEATURE_FILE
    )

    dataframe = prepare_dataset(
        dataframe
    )

    print(
        f"Real-time rows : "
        f"{len(dataframe):,}"
    )

    print(
        f"Real-time end  : "
        f"{dataframe['timestamp'].max()}"
    )

    return dataframe


# ============================================================
# Real-Time History Coverage
# ============================================================

def has_sufficient_realtime_history(
    city_data,
):
    """
    Determine whether real-time data covers enough elapsed
    time to support 24-hour autoregressive features.
    """

    if city_data is None:
        return False

    if city_data.empty:
        return False

    city_data = (
        city_data
        .sort_values(
            "timestamp"
        )
    )

    latest = (
        city_data[
            "timestamp"
        ].max()
    )

    earliest = (
        city_data[
            "timestamp"
        ].min()
    )

    coverage_hours = (
        latest
        - earliest
    ).total_seconds() / 3600

    return (
        coverage_hours
        >= MINIMUM_HISTORY_HOURS
    )


# ============================================================
# Latest Row Validation
# ============================================================

def latest_row_is_model_ready(
    city_data,
    expected_columns,
):
    """
    Verify latest row contains the model's required inputs.

    Some future recursive AQI fields will be recalculated,
    but current meteorological/pollutant variables need to
    exist before forecasting starts.
    """

    if city_data.empty:
        return False

    latest = (
        city_data
        .sort_values(
            "timestamp"
        )
        .iloc[-1]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column
        not in latest.index
    ]

    if missing_columns:

        return False

    important_columns = [
        column
        for column in expected_columns
        if column
        not in [
            "aqi_lag_1",
            "aqi_lag_3",
            "aqi_lag_6",
            "aqi_lag_24",
            "aqi_roll_3",
            "aqi_roll_6",
            "aqi_roll_24",
        ]
    ]

    for column in important_columns:

        value = latest[
            column
        ]

        if pd.isna(
            value
        ):

            return False

    return True


# ============================================================
# Source Selection
# ============================================================

def select_city_forecast_source(
    city,
    realtime_df,
    historical_df,
    expected_columns,
):
    """
    Prefer real-time features when adequate history exists.

    Otherwise fall back to historical data.

    Returns
    -------
    city_data
    source_name
    """

    if realtime_df is not None:

        realtime_city = (
            realtime_df[
                realtime_df[
                    "city"
                ]
                == city
            ]
            .copy()
        )

        if (
            not realtime_city.empty
            and
            has_sufficient_realtime_history(
                realtime_city
            )
            and
            latest_row_is_model_ready(
                realtime_city,
                expected_columns,
            )
        ):

            return (
                realtime_city,
                "REALTIME",
            )

    if historical_df is not None:

        historical_city = (
            historical_df[
                historical_df[
                    "city"
                ]
                == city
            ]
            .copy()
        )

        if not historical_city.empty:

            return (
                historical_city,
                "HISTORICAL_FALLBACK",
            )

    raise RuntimeError(
        f"No usable forecasting data "
        f"is available for {city}."
    )


# ============================================================
# Hourly AQI History
# ============================================================

def build_hourly_aqi_history(
    city_data,
):
    """
    Convert available AQI observations into an hourly series.

    Real-time features may be stored at 3-hour intervals.
    The production model, however, was trained with hourly lag
    semantics:

        lag_1  = 1 hour
        lag_3  = 3 hours
        lag_6  = 6 hours
        lag_24 = 24 hours

    Therefore the observed series is reindexed hourly and
    forward-filled between available observations.
    """

    history = (
        city_data[
            [
                "timestamp",
                "us_aqi",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=[
                "timestamp"
            ],
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )[
            "us_aqi"
        ]
        .astype(float)
    )

    if history.empty:

        raise ValueError(
            "Cannot create AQI history "
            "from an empty series."
        )

    full_index = pd.date_range(
        start=history.index.min(),
        end=history.index.max(),
        freq="1h",
    )

    history = (
        history
        .reindex(
            full_index
        )
        .ffill()
    )

    history.index.name = (
        "timestamp"
    )

    return history


# ============================================================
# Historical AQI Lookup
# ============================================================

def get_aqi_at_or_before(
    history,
    timestamp,
):
    """
    Retrieve AQI at an exact timestamp or nearest prior value.
    """

    if history.empty:

        raise ValueError(
            "AQI history is empty."
        )

    if timestamp in history.index:

        return float(
            history.loc[
                timestamp
            ]
        )

    available = history[
        history.index
        <= timestamp
    ]

    if available.empty:

        return float(
            history.iloc[0]
        )

    return float(
        available.iloc[-1]
    )


# ============================================================
# Rolling AQI Mean
# ============================================================

def get_rolling_mean(
    history,
    forecast_timestamp,
    hours,
):
    """
    Calculate historical hourly AQI average over a window
    ending immediately before the forecast timestamp.
    """

    start_time = (
        forecast_timestamp
        - pd.Timedelta(
            hours=hours
        )
    )

    end_time = (
        forecast_timestamp
        - pd.Timedelta(
            hours=1
        )
    )

    values = history[
        (
            history.index
            >= start_time
        )
        &
        (
            history.index
            <= end_time
        )
    ]

    if values.empty:

        return float(
            history.iloc[-1]
        )

    return float(
        values.mean()
    )


# ============================================================
# Update Time Features
# ============================================================

def update_time_features(
    row,
    timestamp,
):
    """
    Update future calendar features.
    """

    if "hour" in row.index:
        row["hour"] = timestamp.hour

    if "day" in row.index:
        row["day"] = timestamp.day

    if "month" in row.index:
        row["month"] = timestamp.month

    if "year" in row.index:
        row["year"] = timestamp.year

    if "day_of_week" in row.index:

        row[
            "day_of_week"
        ] = timestamp.day_name()

    if "is_weekend" in row.index:

        row[
            "is_weekend"
        ] = int(
            timestamp.dayofweek
            >= 5
        )

    if "season" in row.index:

        row[
            "season"
        ] = get_season(
            timestamp.month
        )

    return row


# ============================================================
# Update AQI Autoregressive Features
# ============================================================

def update_aqi_features(
    row,
    history,
    forecast_timestamp,
):
    """
    Rebuild AQI lag/rolling features using hourly semantics.
    """

    # AQI known immediately before prediction.
    current_timestamp = (
        forecast_timestamp
        - pd.Timedelta(
            hours=1
        )
    )

    current_aqi = (
        get_aqi_at_or_before(
            history,
            current_timestamp,
        )
    )

    if "us_aqi" in row.index:

        row[
            "us_aqi"
        ] = current_aqi

    lag_mapping = {
        "aqi_lag_1": 1,
        "aqi_lag_3": 3,
        "aqi_lag_6": 6,
        "aqi_lag_24": 24,
    }

    for (
        column,
        hours,
    ) in lag_mapping.items():

        if column not in row.index:
            continue

        lookup_time = (
            forecast_timestamp
            - pd.Timedelta(
                hours=hours
            )
        )

        row[
            column
        ] = (
            get_aqi_at_or_before(
                history,
                lookup_time,
            )
        )

    rolling_mapping = {
        "aqi_roll_3": 3,
        "aqi_roll_6": 6,
        "aqi_roll_24": 24,
    }

    for (
        column,
        hours,
    ) in rolling_mapping.items():

        if column not in row.index:
            continue

        row[
            column
        ] = (
            get_rolling_mean(
                history,
                forecast_timestamp,
                hours,
            )
        )

    return row


# ============================================================
# Extend Recursive History
# ============================================================

def extend_recursive_history(
    history,
    previous_timestamp,
    forecast_timestamp,
    previous_aqi,
    predicted_aqi,
):
    """
    Add synthetic hourly AQI points between 3-hour forecasts.

    Example
    -------
    Known/predicted AQI at 20:00.

    Forecast is for 23:00.

    21:00 and 22:00 use persistence of the most recent AQI.
    23:00 receives the newly predicted AQI.

    This keeps lag_1, lag_3, lag_6 and lag_24 aligned with the
    hourly training semantics.
    """

    history = history.copy()

    timestamp = (
        previous_timestamp
        + pd.Timedelta(
            hours=1
        )
    )

    while (
        timestamp
        < forecast_timestamp
    ):

        history.loc[
            timestamp
        ] = float(
            previous_aqi
        )

        timestamp = (
            timestamp
            + pd.Timedelta(
                hours=1
            )
        )

    history.loc[
        forecast_timestamp
    ] = float(
        predicted_aqi
    )

    history = (
        history
        .sort_index()
    )

    return history


# ============================================================
# Prepare Model Input
# ============================================================

def prepare_model_input(
    row,
    expected_columns,
):
    """
    Create exact raw feature structure expected by pipeline.
    """

    dataframe = pd.DataFrame(
        [
            row
        ]
    )

    dataframe = dataframe.drop(
        columns=[
            column
            for column
            in NON_MODEL_COLUMNS
            if column
            in dataframe.columns
        ],
        errors="ignore",
    )

    missing_columns = [
        column
        for column
        in expected_columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Forecast row is missing model features:\n"
            f"{missing_columns}"
        )

    dataframe = dataframe[
        expected_columns
    ]

    return dataframe


# ============================================================
# Forecast Single City
# ============================================================

def forecast_city(
    model,
    city_data,
    city,
    expected_columns,
    production_info,
    source_name,
):
    """
    Generate a recursive 72-hour AQI forecast for one city.
    """

    city_data = (
        city_data
        .sort_values(
            "timestamp"
        )
        .copy()
    )

    if city_data.empty:

        raise ValueError(
            f"No data found for {city}."
        )

    latest_row = (
        city_data
        .iloc[-1]
        .copy()
    )

    latest_timestamp = pd.Timestamp(
        latest_row[
            "timestamp"
        ]
    )

    history = (
        build_hourly_aqi_history(
            city_data
        )
    )

    coverage_hours = (
        history.index.max()
        - history.index.min()
    ).total_seconds() / 3600

    if (
        coverage_hours
        < MINIMUM_HISTORY_HOURS
    ):

        raise ValueError(
            f"{city} does not contain "
            f"{MINIMUM_HISTORY_HOURS} hours "
            "of usable AQI history."
        )

    forecast_rows = []

    previous_timestamp = (
        latest_timestamp
    )

    previous_aqi = float(
        history.iloc[-1]
    )

    for step in range(
        1,
        FORECAST_STEPS + 1,
    ):

        hours_ahead = (
            step
            * FORECAST_INTERVAL_HOURS
        )

        forecast_timestamp = (
            latest_timestamp
            + pd.Timedelta(
                hours=hours_ahead
            )
        )

        # ----------------------------------------------------
        # Add persistence values for intermediate hours.
        # ----------------------------------------------------

        intermediate_timestamp = (
            previous_timestamp
            + pd.Timedelta(
                hours=1
            )
        )

        while (
            intermediate_timestamp
            < forecast_timestamp
        ):

            history.loc[
                intermediate_timestamp
            ] = previous_aqi

            intermediate_timestamp = (
                intermediate_timestamp
                + pd.Timedelta(
                    hours=1
                )
            )

        history = (
            history
            .sort_index()
        )

        # ----------------------------------------------------
        # Start from latest known meteorological/pollutant
        # conditions.
        # ----------------------------------------------------

        future_row = (
            latest_row.copy()
        )

        future_row[
            "timestamp"
        ] = forecast_timestamp

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        future_row = (
            update_time_features(
                future_row,
                forecast_timestamp,
            )
        )

        # ----------------------------------------------------
        # Recursive AQI features
        # ----------------------------------------------------

        future_row = (
            update_aqi_features(
                future_row,
                history,
                forecast_timestamp,
            )
        )

        # ----------------------------------------------------
        # Exact model input
        # ----------------------------------------------------

        X_future = (
            prepare_model_input(
                future_row,
                expected_columns,
            )
        )

        # ----------------------------------------------------
        # Predict t + 3h
        # ----------------------------------------------------

        prediction = model.predict(
            X_future
        )

        predicted_aqi = float(
            prediction[0]
        )

        predicted_aqi = max(
            0.0,
            predicted_aqi,
        )

        predicted_aqi = min(
            500.0,
            predicted_aqi,
        )

        # ----------------------------------------------------
        # Add recursive prediction to hourly history
        # ----------------------------------------------------

        history.loc[
            forecast_timestamp
        ] = predicted_aqi

        history = (
            history
            .sort_index()
        )

        # ----------------------------------------------------
        # Update recursive state
        # ----------------------------------------------------

        latest_row = (
            future_row.copy()
        )

        latest_row[
            "us_aqi"
        ] = predicted_aqi

        latest_row[
            "timestamp"
        ] = forecast_timestamp

        previous_timestamp = (
            forecast_timestamp
        )

        previous_aqi = (
            predicted_aqi
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        forecast_rows.append(
            {
                "timestamp":
                    forecast_timestamp,

                "city":
                    city,

                "forecast_step":
                    step,

                "hours_ahead":
                    hours_ahead,

                "predicted_aqi":
                    round(
                        predicted_aqi,
                        2,
                    ),

                "aqi_category":
                    get_aqi_category(
                        predicted_aqi
                    ),

                "alert_level":
                    get_alert_level(
                        predicted_aqi
                    ),

                "health_guidance":
                    get_health_guidance(
                        predicted_aqi
                    ),

                "model":
                    production_info[
                        "model_name"
                    ],

                "model_version":
                    production_info[
                        "version"
                    ],

                "data_source":
                    source_name,

                "forecast_method":
                    "recursive_persistence",
            }
        )

    return pd.DataFrame(
        forecast_rows
    )


# ============================================================
# Generate All Forecasts
# ============================================================

def generate_3day_forecast(
    model,
    production_info,
    realtime_df,
    historical_df,
):
    """
    Generate 72-hour forecasts for all supported cities.
    """

    print("\n" + "=" * 70)
    print("GENERATING PRODUCTION 3-DAY AQI FORECAST")
    print("=" * 70)

    expected_columns = (
        get_model_feature_columns(
            model
        )
    )

    print(
        f"\nModel            : "
        f"{production_info['model_name']}"
    )

    print(
        f"Version          : "
        f"{production_info['version']}"
    )

    print(
        f"Model features   : "
        f"{len(expected_columns)}"
    )

    print(
        f"Forecast horizon : "
        f"{FORECAST_HOURS} hours"
    )

    print(
        f"Forecast interval: "
        f"{FORECAST_INTERVAL_HOURS} hours"
    )

    print(
        f"Points per city  : "
        f"{FORECAST_STEPS}"
    )

    forecasts = []

    skipped_cities = []

    for city in SUPPORTED_CITIES:

        print(
            "\n" + "-" * 60
        )

        print(
            f"Forecasting: "
            f"{city}"
        )

        try:

            (
                city_data,
                source_name,
            ) = (
                select_city_forecast_source(
                    city=city,
                    realtime_df=(
                        realtime_df
                    ),
                    historical_df=(
                        historical_df
                    ),
                    expected_columns=(
                        expected_columns
                    ),
                )
            )

            print(
                f"Data source: "
                f"{source_name}"
            )

            print(
                f"Latest observation: "
                f"{city_data['timestamp'].max()}"
            )

            city_forecast = (
                forecast_city(
                    model=model,
                    city_data=city_data,
                    city=city,
                    expected_columns=(
                        expected_columns
                    ),
                    production_info=(
                        production_info
                    ),
                    source_name=(
                        source_name
                    ),
                )
            )

            forecasts.append(
                city_forecast
            )

            print(
                f"Created "
                f"{len(city_forecast)} "
                "forecast points."
            )

        except Exception as error:

            print(
                f"Forecast failed for "
                f"{city}: {error}"
            )

            skipped_cities.append(
                city
            )

    if not forecasts:

        raise RuntimeError(
            "No city forecasts could be generated."
        )

    result = pd.concat(
        forecasts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if skipped_cities:

        print(
            "\nCities skipped:"
        )

        for city in skipped_cities:

            print(
                f"  - {city}"
            )

    return result


# ============================================================
# Save Forecast
# ============================================================

def save_forecast(
    forecast_df,
):
    """
    Save 3-day forecast CSV.
    """

    forecast_df.to_csv(
        FORECAST_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FORECAST SAVED")
    print("=" * 70)

    print(
        f"\nFile:\n"
        f"{FORECAST_FILE}"
    )

    print(
        f"\nRows: "
        f"{len(forecast_df):,}"
    )


# ============================================================
# Summary
# ============================================================

def display_summary(
    forecast_df,
    production_info,
):
    """
    Display production forecast summary.
    """

    print("\n" + "=" * 70)
    print("3-DAY AQI FORECAST SUMMARY")
    print("=" * 70)

    print(
        f"\nProduction Model : "
        f"{production_info['model_name']}"
    )

    print(
        f"Model Version    : "
        f"{production_info['version']}"
    )

    print(
        f"Forecast Days    : "
        f"{FORECAST_DAYS}"
    )

    print(
        f"Forecast Hours   : "
        f"{FORECAST_HOURS}"
    )

    print(
        f"Interval         : "
        f"{FORECAST_INTERVAL_HOURS} hours"
    )

    print(
        f"Cities           : "
        f"{forecast_df['city'].nunique()}"
    )

    print(
        f"Predictions      : "
        f"{len(forecast_df):,}"
    )

    print(
        f"Forecast Start   : "
        f"{forecast_df['timestamp'].min()}"
    )

    print(
        f"Forecast End     : "
        f"{forecast_df['timestamp'].max()}"
    )

    print(
        f"Average AQI      : "
        f"{forecast_df['predicted_aqi'].mean():.2f}"
    )

    print(
        f"Maximum AQI      : "
        f"{forecast_df['predicted_aqi'].max():.2f}"
    )

    print(
        f"Minimum AQI      : "
        f"{forecast_df['predicted_aqi'].min():.2f}"
    )

    hazardous = (
        forecast_df[
            forecast_df[
                "predicted_aqi"
            ]
            > 300
        ]
    )

    print(
        f"\nHazardous predictions (>300): "
        f"{len(hazardous)}"
    )

    print(
        "\nData-source usage:"
    )

    print(
        forecast_df[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )

    city_summary = (
        forecast_df
        .groupby(
            "city"
        )
        .agg(
            average_aqi=(
                "predicted_aqi",
                "mean",
            ),
            maximum_aqi=(
                "predicted_aqi",
                "max",
            ),
            minimum_aqi=(
                "predicted_aqi",
                "min",
            ),
        )
        .round(2)
        .reset_index()
    )

    print(
        "\nForecast by city:"
    )

    print(
        city_summary.to_string(
            index=False
        )
    )

    print(
        "\nFirst 20 forecast rows:"
    )

    display_columns = [
        "timestamp",
        "city",
        "forecast_step",
        "hours_ahead",
        "predicted_aqi",
        "aqi_category",
        "alert_level",
        "model",
        "model_version",
        "data_source",
    ]

    print(
        forecast_df[
            display_columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PRODUCTION 3-DAY AQI FORECASTING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # Production Model
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 1: LOADING PRODUCTION MODEL")
    print("-" * 70)

    (
        model,
        production_info,
    ) = (
        load_production_model()
    )

    # ========================================================
    # STEP 2
    # Expected Features
    # ========================================================

    expected_columns = (
        get_model_feature_columns(
            model
        )
    )

    print(
        f"\nProduction model expects "
        f"{len(expected_columns)} "
        "raw features."
    )

    # ========================================================
    # STEP 3
    # Real-Time Data
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 2: LOADING REAL-TIME FEATURES")
    print("-" * 70)

    try:

        realtime_df = (
            load_realtime_dataset()
        )

    except Exception as error:

        print(
            "\nReal-time feature loading failed:"
        )

        print(
            error
        )

        realtime_df = None

    # ========================================================
    # STEP 4
    # Historical Fallback
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 3: LOADING HISTORICAL FALLBACK")
    print("-" * 70)

    try:

        historical_df = (
            load_historical_dataset()
        )

    except Exception as error:

        print(
            "\nHistorical dataset loading failed:"
        )

        print(
            error
        )

        historical_df = None

    if (
        realtime_df is None
        and
        historical_df is None
    ):

        raise RuntimeError(
            "Neither real-time nor historical "
            "forecast data is available."
        )

    # ========================================================
    # STEP 5
    # Forecast
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 4: GENERATING 3-DAY FORECAST")
    print("-" * 70)

    forecast_df = (
        generate_3day_forecast(
            model=model,
            production_info=(
                production_info
            ),
            realtime_df=(
                realtime_df
            ),
            historical_df=(
                historical_df
            ),
        )
    )

    # ========================================================
    # STEP 6
    # Save
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 5: SAVING FORECAST")
    print("-" * 70)

    save_forecast(
        forecast_df
    )

    # ========================================================
    # STEP 7
    # Summary
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 6: FORECAST SUMMARY")
    print("-" * 70)

    display_summary(
        forecast_df=(
            forecast_df
        ),
        production_info=(
            production_info
        ),
    )

    # ========================================================
    # Complete
    # ========================================================

    print("\n" + "=" * 70)
    print("3-DAY AQI FORECAST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nForecast file:\n"
        f"{FORECAST_FILE}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()