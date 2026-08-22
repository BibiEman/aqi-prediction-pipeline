"""
realtime_features.py

Build model-compatible real-time AQI features.

Workflow
--------
Raw hourly observations
        ↓
Normalize city names
        ↓
Rename API pollutant columns
        ↓
Calculate US AQI
        ↓
Aggregate into 3-hour city buckets
        ↓
Create time features
        ↓
Create lag features
        ↓
Create rolling features
        ↓
Mark model-ready rows
        ↓
Save real-time feature dataset

Input
-----
data/raw/aqi_dataset.csv

Output
------
data/realtime/realtime_features.csv
"""

from pathlib import Path
import unicodedata

import pandas as pd

from src.aqi_calculation.calculate_aqi import (
    calculate_us_aqi,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aqi_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "realtime"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "realtime_features.csv"
)


# ============================================================
# Configuration
# ============================================================

RESAMPLE_INTERVAL = "3h"


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
# Normalize City
# ============================================================

def normalize_city_name(city):
    """
    Normalize API city names.

    Example:
        Siālkot -> Sialkot
    """

    if pd.isna(city):
        return city

    city = str(city).strip()

    normalized = unicodedata.normalize(
        "NFKD",
        city,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    lookup = {
        item.lower(): item
        for item in SUPPORTED_CITIES
    }

    return lookup.get(
        normalized.lower(),
        normalized,
    )


# ============================================================
# Load Raw Data
# ============================================================

def load_realtime_data():
    """
    Load cumulative real-time observations.
    """

    print("\n" + "=" * 70)
    print("LOADING RAW REAL-TIME DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Real-time raw dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "Real-time raw dataset is empty."
        )

    required_columns = [
        "timestamp",
        "city",
        "pm2_5",
        "pm10",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required real-time columns:\n"
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps > 0:

        print(
            f"\nRemoving {invalid_timestamps} "
            "rows with invalid timestamps..."
        )

        df = df[
            df["timestamp"].notna()
        ].copy()

    df["city"] = (
        df["city"]
        .apply(
            normalize_city_name
        )
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

    print(
        f"\nRows   : {len(df):,}"
    )

    print(
        f"Cities : {df['city'].nunique()}"
    )

    print(
        f"Range  : "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    return df


# ============================================================
# Standardize Column Names
# ============================================================

def standardize_columns(df):
    """
    Rename API columns into training-schema names.
    """

    rename_map = {
        "co":
            "carbon_monoxide",

        "no":
            "nitrogen_monoxide",

        "no2":
            "nitrogen_dioxide",

        "o3":
            "ozone",

        "so2":
            "sulphur_dioxide",

        "nh3":
            "ammonia",

        "aqi":
            "openweather_aqi_index",
    }

    existing_map = {
        source: target
        for source, target
        in rename_map.items()
        if (
            source in df.columns
            and target not in df.columns
        )
    }

    return df.rename(
        columns=existing_map
    )


# ============================================================
# Add Missing Columns
# ============================================================

def add_missing_model_columns(df):
    """
    Add model fields that may be missing from live APIs.
    """

    defaults = {
        "wind_direction": 0.0,
        "precipitation": 0.0,
    }

    for (
        column,
        default_value,
    ) in defaults.items():

        if column not in df.columns:

            df[column] = (
                default_value
            )

    return df


# ============================================================
# Calculate US AQI
# ============================================================

def create_us_aqi(df):
    """
    Calculate US AQI from PM2.5 and PM10.
    """

    print(
        "\nCalculating US AQI..."
    )

    df["us_aqi"] = df.apply(
        lambda row:
        calculate_us_aqi(
            pm2_5=row.get(
                "pm2_5"
            ),
            pm10=row.get(
                "pm10"
            ),
        ),
        axis=1,
    )

    return df


# ============================================================
# 3-Hour Aggregation
# ============================================================

def aggregate_to_three_hours(df):
    """
    Aggregate raw observations into 3-hour buckets.

    This aligns the live feature cadence with the historical
    model-training cadence.

    Numerical values are averaged within each 3-hour period.
    """

    print("\n" + "=" * 70)
    print("AGGREGATING INTO 3-HOUR INTERVALS")
    print("=" * 70)

    # --------------------------------------------------------
    # Numeric features we want to average
    # --------------------------------------------------------

    possible_numeric_columns = [
        "latitude",
        "longitude",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "cloud_cover",
        "visibility",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "ammonia",
        "openweather_aqi_index",
        "us_aqi",
    ]

    numeric_columns = [
        column
        for column
        in possible_numeric_columns
        if column in df.columns
    ]

    # --------------------------------------------------------
    # Ensure numeric
    # --------------------------------------------------------

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Floor timestamp to 3-hour bucket
    # --------------------------------------------------------

    df["bucket_timestamp"] = (
        df["timestamp"]
        .dt.floor(
            RESAMPLE_INTERVAL
        )
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        df.groupby(
            [
                "city",
                "bucket_timestamp",
            ],
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    grouped = grouped.rename(
        columns={
            "bucket_timestamp":
                "timestamp"
        }
    )

    # --------------------------------------------------------
    # Round AQI to realistic value
    # --------------------------------------------------------

    if "us_aqi" in grouped.columns:

        grouped["us_aqi"] = (
            grouped["us_aqi"]
            .round()
        )

    grouped = (
        grouped
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nRaw rows       : "
        f"{len(df):,}"
    )

    print(
        f"3-hour rows    : "
        f"{len(grouped):,}"
    )

    print(
        f"Cities         : "
        f"{grouped['city'].nunique()}"
    )

    print(
        f"3-hour range   : "
        f"{grouped['timestamp'].min()} "
        f"→ "
        f"{grouped['timestamp'].max()}"
    )

    return grouped


# ============================================================
# Time Features
# ============================================================

def create_time_features(df):
    """
    Create calendar features matching training data.
    """

    print(
        "\nCreating time features..."
    )

    df["hour"] = (
        df["timestamp"]
        .dt.hour
    )

    df["day"] = (
        df["timestamp"]
        .dt.day
    )

    df["month"] = (
        df["timestamp"]
        .dt.month
    )

    df["year"] = (
        df["timestamp"]
        .dt.year
    )

    df["day_of_week"] = (
        df["timestamp"]
        .dt.day_name()
    )

    df["is_weekend"] = (
        df["timestamp"]
        .dt.dayofweek
        >= 5
    ).astype(int)

    def get_season(month):

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

    df["season"] = (
        df["month"]
        .apply(
            get_season
        )
    )

    return df


# ============================================================
# Lag Features
# ============================================================

def create_lag_features(df):
    """
    Create AQI lag features on 3-hour observations.

    Interpretation
    --------------
    aqi_lag_1  = 3 hours ago
    aqi_lag_3  = 9 hours ago
    aqi_lag_6  = 18 hours ago
    aqi_lag_24 = 72 hours ago
    """

    print(
        "Creating 3-hour AQI lag features..."
    )

    grouped = (
        df.groupby(
            "city"
        )[
            "us_aqi"
        ]
    )

    df["aqi_lag_1"] = (
        grouped.shift(1)
    )

    df["aqi_lag_3"] = (
        grouped.shift(3)
    )

    df["aqi_lag_6"] = (
        grouped.shift(6)
    )

    df["aqi_lag_24"] = (
        grouped.shift(24)
    )

    return df


# ============================================================
# Rolling Features
# ============================================================

def create_rolling_features(df):
    """
    Create rolling AQI averages using 3-hour observations.

    Interpretation
    --------------
    roll_3  = latest 3 observations  = 9 hours
    roll_6  = latest 6 observations  = 18 hours
    roll_24 = latest 24 observations = 72 hours
    """

    print(
        "Creating 3-hour AQI rolling features..."
    )

    grouped = (
        df.groupby(
            "city"
        )[
            "us_aqi"
        ]
    )

    df["aqi_roll_3"] = (
        grouped.transform(
            lambda series:
            series.rolling(
                window=3,
                min_periods=3,
            ).mean()
        )
    )

    df["aqi_roll_6"] = (
        grouped.transform(
            lambda series:
            series.rolling(
                window=6,
                min_periods=6,
            ).mean()
        )
    )

    df["aqi_roll_24"] = (
        grouped.transform(
            lambda series:
            series.rolling(
                window=24,
                min_periods=24,
            ).mean()
        )
    )

    return df


# ============================================================
# Feature Validation
# ============================================================

def validate_features(df):
    """
    Print real-time feature availability.
    """

    print("\n" + "=" * 70)
    print("REAL-TIME FEATURE VALIDATION")
    print("=" * 70)

    features = [
        "us_aqi",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
    ]

    for feature in features:

        if feature not in df.columns:
            continue

        available = int(
            df[
                feature
            ]
            .notna()
            .sum()
        )

        missing = int(
            df[
                feature
            ]
            .isna()
            .sum()
        )

        print(
            f"{feature:15s} "
            f"available={available:5d} "
            f"missing={missing:5d}"
        )

    return df


# ============================================================
# Model-Ready Rows
# ============================================================

def mark_model_ready_rows(df):
    """
    Mark rows where all required historical AQI features exist.
    """

    required_history_features = [
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
    ]

    ready_mask = (
        df[
            required_history_features
        ]
        .notna()
        .all(
            axis=1
        )
    )

    df["model_ready"] = (
        ready_mask.astype(int)
    )

    return df


# ============================================================
# Latest Observation Summary
# ============================================================

def display_latest_rows(df):
    """
    Display most recent 3-hour feature row per city.
    """

    print("\n" + "=" * 70)
    print("LATEST 3-HOUR FEATURE ROWS")
    print("=" * 70)

    latest = (
        df.sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .groupby(
            "city",
            as_index=False,
        )
        .tail(1)
        .sort_values(
            "city"
        )
    )

    columns = [
        column
        for column in [
            "timestamp",
            "city",
            "us_aqi",
            "aqi_lag_1",
            "aqi_lag_3",
            "aqi_lag_6",
            "aqi_lag_24",
            "model_ready",
        ]
        if column in latest.columns
    ]

    print(
        "\n"
        + latest[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# Save Features
# ============================================================

def save_features(df):
    """
    Save real-time 3-hour feature dataset.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    ready_count = int(
        df[
            "model_ready"
        ].sum()
    )

    print("\n" + "=" * 70)
    print("REAL-TIME FEATURES SAVED")
    print("=" * 70)

    print(
        f"\nFile:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nRows        : "
        f"{len(df):,}"
    )

    print(
        f"Cities      : "
        f"{df['city'].nunique()}"
    )

    print(
        f"Model-ready : "
        f"{ready_count:,}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI 3-HOUR REAL-TIME FEATURE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    df = load_realtime_data()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    df = standardize_columns(
        df
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    df = add_missing_model_columns(
        df
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    df = create_us_aqi(
        df
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    df = aggregate_to_three_hours(
        df
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    df = create_time_features(
        df
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    df = create_lag_features(
        df
    )

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    df = create_rolling_features(
        df
    )

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    df = validate_features(
        df
    )

    # --------------------------------------------------------
    # Step 10
    # --------------------------------------------------------

    df = mark_model_ready_rows(
        df
    )

    # --------------------------------------------------------
    # Step 11
    # --------------------------------------------------------

    display_latest_rows(
        df
    )

    # --------------------------------------------------------
    # Step 12
    # --------------------------------------------------------

    save_features(
        df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3-HOUR REAL-TIME FEATURE PIPELINE COMPLETED")
    print("=" * 70)

    ready_count = int(
        df[
            "model_ready"
        ].sum()
    )

    if ready_count == 0:

        print(
            "\nNo model-ready real-time rows yet."
        )

        print(
            "This is expected until sufficient "
            "3-hour history has accumulated."
        )

        print(
            "\nThe longest required feature is "
            "aqi_lag_24 / aqi_roll_24."
        )

        print(
            "That requires approximately 72 hours "
            "of 3-hour observations per city."
        )

    else:

        print(
            f"\nModel-ready rows: "
            f"{ready_count:,}"
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()