"""
predict.py

3-Day AQI Forecasting Pipeline

Purpose
-------
Generate AQI forecasts for the next 3 days using the trained
LightGBM model.

The model was trained with:

    target_aqi(t) = us_aqi(t + 3 hours)

Therefore forecasts are generated recursively every 3 hours.

Forecast horizon:
    72 hours = 3 days

Forecast interval:
    3 hours

Forecast points per city:
    72 / 3 = 24

For 10 cities:
    24 x 10 = 240 predictions

IMPORTANT
---------
The saved LightGBM .joblib file already contains the complete:

    ColumnTransformer
          +
    OneHotEncoder
          +
    LightGBM model

Therefore categorical features are NOT manually encoded.

Future weather/pollutant values are currently handled using a
persistence assumption: the latest available values are retained.

For a production deployment, replace these persisted values with
future weather and pollution forecasts from an external API.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model_training.config import (
    MODEL_DIR,
    RESULTS_DIR,
)

from src.model_training.utils import (
    load_dataset,
    load_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

BEST_MODEL_NAME = "LightGBM"

FORECAST_DAYS = 3

FORECAST_INTERVAL_HOURS = 3

FORECAST_HOURS = (
    FORECAST_DAYS * 24
)

FORECAST_STEPS = (
    FORECAST_HOURS
    // FORECAST_INTERVAL_HOURS
)


MODEL_PATH = (
    Path(MODEL_DIR)
    / f"{BEST_MODEL_NAME}.joblib"
)


PREDICTION_DIR = (
    Path(RESULTS_DIR)
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
# FEATURE COLUMNS
# ============================================================

AQI_LAG_COLUMNS = [
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",
]


AQI_ROLLING_COLUMNS = [
    "aqi_roll_3",
    "aqi_roll_6",
    "aqi_roll_24",
]


NON_MODEL_COLUMNS = [
    "timestamp",
    "target_aqi",
]


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):
    """
    Convert AQI value into a human-readable category.

    US AQI categories are used.
    """

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# AQI ALERT
# ============================================================

def get_alert_level(aqi):
    """
    Generate a simple alert level.
    """

    if aqi <= 100:
        return "Normal"

    if aqi <= 150:
        return "Caution"

    if aqi <= 200:
        return "Health Alert"

    if aqi <= 300:
        return "Severe Health Alert"

    return "Emergency"


# ============================================================
# SEASON
# ============================================================

def get_season(month):
    """
    Convert month into season.
    """

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


# ============================================================
# LOAD BEST MODEL
# ============================================================

def load_best_model():
    """
    Load trained LightGBM pipeline.
    """

    print("\n" + "=" * 70)
    print("LOADING PRODUCTION MODEL")
    print("=" * 70)

    print(
        f"\nModel : {BEST_MODEL_NAME}"
    )

    print(
        f"Path  : {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nTrained model was not found.\n"
            f"Expected:\n{MODEL_PATH}"
        )

    model = load_model(
        MODEL_PATH
    )

    print(
        "\nProduction model loaded successfully."
    )

    return model


# ============================================================
# GET EXPECTED MODEL FEATURES
# ============================================================

def get_model_feature_columns(model):
    """
    Get the raw input columns expected by the saved pipeline.

    The sklearn pipeline stores the feature names used when
    the preprocessor was fitted.
    """

    if not hasattr(
        model,
        "named_steps",
    ):
        raise RuntimeError(
            "Loaded model is not a sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise RuntimeError(
            "Pipeline does not contain a preprocessor."
        )

    preprocessor = (
        model.named_steps["preprocessor"]
    )

    if hasattr(
        preprocessor,
        "feature_names_in_",
    ):

        return list(
            preprocessor.feature_names_in_
        )

    raise RuntimeError(
        "Could not determine model input features."
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

def prepare_dataset(df):
    """
    Validate and sort the historical dataset.
    """

    required_columns = [
        "timestamp",
        "city",
        "us_aqi",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Dataset is missing required columns:\n"
            f"{missing}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found."
        )

    df = (
        df.sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# UPDATE TIME FEATURES
# ============================================================

def update_time_features(
    row,
    timestamp,
):
    """
    Update calendar features for a future timestamp.
    """

    if "hour" in row.index:

        row["hour"] = (
            timestamp.hour
        )

    if "day" in row.index:

        row["day"] = (
            timestamp.day
        )

    if "month" in row.index:

        row["month"] = (
            timestamp.month
        )

    if "year" in row.index:

        row["year"] = (
            timestamp.year
        )

    if "day_of_week" in row.index:

        row["day_of_week"] = (
            timestamp.day_name()
        )

    if "is_weekend" in row.index:

        row["is_weekend"] = int(
            timestamp.dayofweek >= 5
        )

    if "season" in row.index:

        row["season"] = (
            get_season(
                timestamp.month
            )
        )

    return row


# ============================================================
# SAFE HISTORICAL AQI LOOKUP
# ============================================================

def get_history_value(
    history,
    hours_back,
):
    """
    Return an AQI value from available AQI history.

    The history is hourly historical data followed by recursively
    predicted values.

    When an exact intermediate hourly prediction is unavailable,
    the nearest available previous value is used.
    """

    if not history:

        raise ValueError(
            "AQI history is empty."
        )

    index = (
        len(history)
        - hours_back
    )

    if index < 0:

        return float(
            history[0]
        )

    return float(
        history[index]
    )


# ============================================================
# UPDATE AQI FEATURES
# ============================================================

def update_aqi_features(
    row,
    aqi_history,
):
    """
    Recalculate lag and rolling AQI features.

    aqi_history contains historical and recursively generated AQI
    values.
    """

    if not aqi_history:

        raise ValueError(
            "Cannot calculate AQI features "
            "because AQI history is empty."
        )

    # --------------------------------------------------------
    # Current AQI
    # --------------------------------------------------------

    current_aqi = float(
        aqi_history[-1]
    )

    if "us_aqi" in row.index:

        row["us_aqi"] = (
            current_aqi
        )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    if "aqi_lag_1" in row.index:

        row["aqi_lag_1"] = (
            get_history_value(
                aqi_history,
                1,
            )
        )

    if "aqi_lag_3" in row.index:

        row["aqi_lag_3"] = (
            get_history_value(
                aqi_history,
                3,
            )
        )

    if "aqi_lag_6" in row.index:

        row["aqi_lag_6"] = (
            get_history_value(
                aqi_history,
                6,
            )
        )

    if "aqi_lag_24" in row.index:

        row["aqi_lag_24"] = (
            get_history_value(
                aqi_history,
                24,
            )
        )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    if "aqi_roll_3" in row.index:

        values = (
            aqi_history[-3:]
        )

        row["aqi_roll_3"] = float(
            np.mean(values)
        )

    if "aqi_roll_6" in row.index:

        values = (
            aqi_history[-6:]
        )

        row["aqi_roll_6"] = float(
            np.mean(values)
        )

    if "aqi_roll_24" in row.index:

        values = (
            aqi_history[-24:]
        )

        row["aqi_roll_24"] = float(
            np.mean(values)
        )

    return row


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    row,
    expected_columns,
):
    """
    Convert one future observation into the exact feature
    structure expected by the saved pipeline.
    """

    X = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    X = X.drop(
        columns=[
            column
            for column in NON_MODEL_COLUMNS
            if column in X.columns
        ],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in expected_columns
        if column not in X.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Future forecast row is missing "
            "model features:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Exact training column order
    # --------------------------------------------------------

    X = X[
        expected_columns
    ]

    return X


# ============================================================
# FORECAST ONE CITY
# ============================================================

def forecast_city(
    model,
    city_data,
    city,
    expected_columns,
):
    """
    Generate a recursive 72-hour forecast for one city.

    Forecast interval:
        3 hours

    Total points:
        24
    """

    city_data = (
        city_data
        .sort_values("timestamp")
        .copy()
    )

    if city_data.empty:

        raise ValueError(
            f"No historical data found for {city}."
        )

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    latest_row = (
        city_data
        .iloc[-1]
        .copy()
    )

    latest_timestamp = (
        latest_row["timestamp"]
    )

    # --------------------------------------------------------
    # Hourly historical AQI
    # --------------------------------------------------------

    aqi_history = (
        city_data[
            "us_aqi"
        ]
        .astype(float)
        .tolist()
    )

    if len(aqi_history) < 24:

        raise ValueError(
            f"{city} does not contain enough "
            "historical AQI observations."
        )

    forecast_rows = []

    # --------------------------------------------------------
    # Recursive forecast
    # --------------------------------------------------------

    for step in range(
        1,
        FORECAST_STEPS + 1,
    ):

        forecast_timestamp = (
            latest_timestamp
            + pd.Timedelta(
                hours=(
                    step
                    * FORECAST_INTERVAL_HOURS
                )
            )
        )

        future_row = (
            latest_row.copy()
        )

        # ----------------------------------------------------
        # Update calendar features
        # ----------------------------------------------------

        future_row = update_time_features(
            future_row,
            forecast_timestamp,
        )

        # ----------------------------------------------------
        # Update AQI lag/rolling features
        # ----------------------------------------------------

        future_row = update_aqi_features(
            future_row,
            aqi_history,
        )

        # ----------------------------------------------------
        # Model input
        # ----------------------------------------------------

        X_future = prepare_model_input(
            future_row,
            expected_columns,
        )

        # ----------------------------------------------------
        # Predict AQI +3h
        # ----------------------------------------------------

        prediction = model.predict(
            X_future
        )

        predicted_aqi = float(
            prediction[0]
        )

        # AQI cannot be negative
        predicted_aqi = max(
            0.0,
            predicted_aqi,
        )

        # ----------------------------------------------------
        # Save prediction in recursive history
        # ----------------------------------------------------

        aqi_history.append(
            predicted_aqi
        )

        # ----------------------------------------------------
        # Update base row for following forecast
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
                    (
                        step
                        * FORECAST_INTERVAL_HOURS
                    ),

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

                "model":
                    BEST_MODEL_NAME,
            }
        )

    return pd.DataFrame(
        forecast_rows
    )


# ============================================================
# FORECAST ALL CITIES
# ============================================================

def generate_3day_forecast(
    model,
    df,
):
    """
    Generate the 3-day forecast for every available city.
    """

    print("\n" + "=" * 70)
    print("GENERATING 3-DAY AQI FORECAST")
    print("=" * 70)

    expected_columns = (
        get_model_feature_columns(
            model
        )
    )

    print(
        f"\nModel expects "
        f"{len(expected_columns)} features."
    )

    cities = sorted(
        df["city"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Cities to forecast: "
        f"{len(cities)}"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HOURS} hours"
    )

    print(
        f"Forecast interval: "
        f"{FORECAST_INTERVAL_HOURS} hours"
    )

    print(
        f"Predictions per city: "
        f"{FORECAST_STEPS}"
    )

    forecasts = []

    for city in cities:

        print("\n" + "-" * 60)

        print(
            f"Forecasting: {city}"
        )

        city_data = (
            df[
                df["city"] == city
            ]
            .copy()
        )

        city_forecast = forecast_city(
            model=model,
            city_data=city_data,
            city=city,
            expected_columns=expected_columns,
        )

        forecasts.append(
            city_forecast
        )

        print(
            f"Created "
            f"{len(city_forecast)} "
            "forecast points."
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
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SAVE FORECAST
# ============================================================

def save_forecast(
    forecast_df,
):
    """
    Save future forecast.
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
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    forecast_df,
):
    """
    Display forecast summary.
    """

    print("\n" + "=" * 70)
    print("3-DAY AQI FORECAST SUMMARY")
    print("=" * 70)

    print(
        f"\nModel            : "
        f"{BEST_MODEL_NAME}"
    )

    print(
        f"Forecast days    : "
        f"{FORECAST_DAYS}"
    )

    print(
        f"Forecast hours   : "
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
        f"Forecast start   : "
        f"{forecast_df['timestamp'].min()}"
    )

    print(
        f"Forecast end     : "
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

    # --------------------------------------------------------
    # Hazardous forecasts
    # --------------------------------------------------------

    hazardous = forecast_df[
        forecast_df[
            "predicted_aqi"
        ] > 300
    ]

    print(
        f"\nHazardous predictions (>300): "
        f"{len(hazardous)}"
    )

    # --------------------------------------------------------
    # City summary
    # --------------------------------------------------------

    city_summary = (
        forecast_df
        .groupby("city")
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

    print(
        forecast_df.head(
            20
        ).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("3-DAY AQI FORECASTING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 1: LOADING HISTORICAL DATA")
    print("-" * 70)

    df = load_dataset()

    df = prepare_dataset(
        df
    )

    print(
        f"\nHistorical rows: "
        f"{len(df):,}"
    )

    print(
        f"Cities: "
        f"{df['city'].nunique()}"
    )

    print(
        f"Latest observation: "
        f"{df['timestamp'].max()}"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 2: LOADING LIGHTGBM MODEL")
    print("-" * 70)

    model = load_best_model()

    # ========================================================
    # STEP 3
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 3: GENERATING 3-DAY FORECAST")
    print("-" * 70)

    forecast_df = generate_3day_forecast(
        model=model,
        df=df,
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 4: SAVING FORECAST")
    print("-" * 70)

    save_forecast(
        forecast_df
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 5: FORECAST SUMMARY")
    print("-" * 70)

    display_summary(
        forecast_df
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("3-DAY AQI FORECAST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nForecast file:\n"
        f"{FORECAST_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()