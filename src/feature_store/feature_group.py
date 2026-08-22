"""
feature_group.py

Online Hopsworks Feature Store integration
for real-time AQI features.

Architecture
------------
Historical training:
    aqi_features v1
        -> offline training/evaluation

Real-time serving:
    aqi_realtime_features v2
        -> online serving

Input
-----
data/realtime/realtime_features.csv

Important
---------
The local Windows client previously failed while writing
to Hopsworks offline Delta/HDFS storage.

Therefore this module writes real-time features using:

    storage="online"

The online Feature Group stores the latest feature vector
for each city.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_store.hopsworks_config import (
    get_feature_store,
    REALTIME_FEATURE_GROUP_NAME,
    REALTIME_FEATURE_GROUP_VERSION,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REALTIME_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / "realtime_features.csv"
)


# ============================================================
# Feature Group Configuration
# ============================================================

# Online serving should expose one current row per city.
PRIMARY_KEY = [
    "city",
]

EVENT_TIME = "timestamp"


# ============================================================
# Feature Schema
# ============================================================

FEATURE_COLUMNS = [
    "timestamp",
    "city",

    "latitude",
    "longitude",

    "pm2_5",
    "pm10",

    "carbon_monoxide",
    "nitrogen_monoxide",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "ammonia",

    "openweather_aqi_index",

    "temperature",
    "humidity",
    "pressure",

    "wind_speed",
    "wind_direction",

    "precipitation",
    "cloud_cover",
    "visibility",

    "us_aqi",

    "hour",
    "day",
    "month",
    "year",

    "day_of_week",
    "is_weekend",
    "season",

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",

    "aqi_roll_3",
    "aqi_roll_6",
    "aqi_roll_24",

    "model_ready",
]


NUMERIC_COLUMNS = [
    "latitude",
    "longitude",

    "pm2_5",
    "pm10",

    "carbon_monoxide",
    "nitrogen_monoxide",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "ammonia",

    "openweather_aqi_index",

    "temperature",
    "humidity",
    "pressure",

    "wind_speed",
    "wind_direction",

    "precipitation",
    "cloud_cover",
    "visibility",

    "us_aqi",

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",

    "aqi_roll_3",
    "aqi_roll_6",
    "aqi_roll_24",
]


INTEGER_COLUMNS = [
    "hour",
    "day",
    "month",
    "year",
    "is_weekend",
    "model_ready",
]


STRING_COLUMNS = [
    "city",
    "day_of_week",
    "season",
]


# ============================================================
# Load Local Features
# ============================================================

def load_realtime_features():
    """
    Load the generated real-time 3-hour features.
    """

    print("\n" + "=" * 70)
    print("LOADING REAL-TIME FEATURES")
    print("=" * 70)

    if not REALTIME_FEATURE_FILE.exists():

        raise FileNotFoundError(
            "Real-time feature file was not found.\n\n"
            f"Expected:\n{REALTIME_FEATURE_FILE}\n\n"
            "Run first:\n"
            "python -m src.realtime.realtime_features"
        )

    df = pd.read_csv(
        REALTIME_FEATURE_FILE
    )

    if df.empty:

        raise ValueError(
            "Real-time feature file is empty."
        )

    required_columns = [
        "timestamp",
        "city",
        "us_aqi",
        "model_ready",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required feature columns:\n"
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid = int(
        df["timestamp"]
        .isna()
        .sum()
    )

    if invalid > 0:

        print(
            f"\nRemoving {invalid} invalid "
            "timestamp rows..."
        )

        df = df[
            df["timestamp"].notna()
        ].copy()

    if df.empty:

        raise ValueError(
            "No valid real-time rows remain."
        )

    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
    )

    print(
        f"\nRows available : {len(df):,}"
    )

    print(
        f"Cities         : {df['city'].nunique()}"
    )

    print(
        f"Date range     : "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Model-ready    : "
        f"{int(df['model_ready'].sum())}"
    )

    return df


# ============================================================
# Latest Feature Vector Per City
# ============================================================

def select_latest_city_features(
    df,
):
    """
    Select the newest 3-hour feature row for each city.

    Online Feature Store purpose:
        one current feature vector per city.
    """

    print("\n" + "=" * 70)
    print("SELECTING LATEST CITY FEATURES")
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
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nRows selected : {len(latest)}"
    )

    print(
        f"Cities        : "
        f"{latest['city'].nunique()}"
    )

    return latest


# ============================================================
# Prepare DataFrame
# ============================================================

def prepare_online_dataframe(
    df,
):
    """
    Prepare latest real-time rows for Hopsworks.
    """

    print("\n" + "=" * 70)
    print("PREPARING ONLINE FEATURE DATA")
    print("=" * 70)

    df = df.copy()

    available_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "\nConfigured columns not currently available:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

    df = df[
        available_columns
    ].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "city",
        ]
    )

    df["timestamp"] = (
        df["timestamp"]
        .astype(
            "datetime64[us]"
        )
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Integers
    # --------------------------------------------------------

    for column in INTEGER_COLUMNS:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Strings
    # --------------------------------------------------------

    for column in STRING_COLUMNS:

        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Invalid infinities
    # --------------------------------------------------------

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # One row per online primary key
    # --------------------------------------------------------

    df = (
        df.drop_duplicates(
            subset=PRIMARY_KEY,
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nPrepared rows    : {len(df)}"
    )

    print(
        f"Prepared columns : {len(df.columns)}"
    )

    return df


# ============================================================
# Get / Create Online Feature Group
# ============================================================

def get_online_feature_group(
    feature_store,
):
    """
    Return a valid online FeatureGroup object.

    Important:
    get_feature_group() may return None if the requested
    version does not exist, so None must be checked explicitly.
    """

    print("\n" + "=" * 70)
    print("HOPSWORKS ONLINE FEATURE GROUP")
    print("=" * 70)

    feature_group = None

    # --------------------------------------------------------
    # Try existing v2
    # --------------------------------------------------------

    try:

        feature_group = (
            feature_store
            .get_feature_group(
                name=REALTIME_FEATURE_GROUP_NAME,
                version=REALTIME_FEATURE_GROUP_VERSION,
            )
        )

    except Exception as error:

        print(
            "\nExisting Feature Group lookup "
            "did not succeed."
        )

        print(
            f"Reason: {error}"
        )

    # --------------------------------------------------------
    # Existing
    # --------------------------------------------------------

    if feature_group is not None:

        print(
            "\nExisting online Feature Group found."
        )

    # --------------------------------------------------------
    # Create lazy metadata object
    # --------------------------------------------------------

    else:

        print(
            "\nOnline Feature Group v2 "
            "does not exist yet."
        )

        print(
            "Creating Feature Group configuration..."
        )

        feature_group = (
            feature_store
            .get_or_create_feature_group(
                name=REALTIME_FEATURE_GROUP_NAME,

                version=REALTIME_FEATURE_GROUP_VERSION,

                description=(
                    "Latest real-time AQI, weather, "
                    "pollutant, lag and rolling features "
                    "for online AQI inference."
                ),

                primary_key=PRIMARY_KEY,

                event_time=EVENT_TIME,

                online_enabled=True,
            )
        )

    if feature_group is None:

        raise RuntimeError(
            "Hopsworks did not return a valid "
            "FeatureGroup object."
        )

    print(
        f"\nFeature Group : "
        f"{REALTIME_FEATURE_GROUP_NAME}"
    )

    print(
        f"Version       : "
        f"{REALTIME_FEATURE_GROUP_VERSION}"
    )

    print(
        f"Primary Key   : {PRIMARY_KEY}"
    )

    print(
        f"Event Time    : {EVENT_TIME}"
    )

    print(
        "Online Store : Enabled"
    )

    print(
        f"Object Type  : "
        f"{type(feature_group).__name__}"
    )

    return feature_group


# ============================================================
# Schema Alignment
# ============================================================

def align_with_existing_schema(
    dataframe,
    feature_group,
):
    """
    If v2 already has a persisted schema, align the
    DataFrame exactly with it.

    If v2 is brand new, leave the DataFrame unchanged
    so Hopsworks can infer its initial schema.
    """

    try:

        features = feature_group.features

    except Exception:

        return dataframe

    if not features:

        print(
            "\nFeature Group has no persisted schema yet."
        )

        print(
            "Schema will be inferred from this first insert."
        )

        return dataframe

    schema_columns = [
        feature.name
        for feature in features
        if getattr(
            feature,
            "name",
            None,
        )
    ]

    if not schema_columns:

        return dataframe

    print(
        f"\nExisting v2 schema columns: "
        f"{len(schema_columns)}"
    )

    local_lookup = {
        column.lower(): column
        for column in dataframe.columns
    }

    schema_lookup = {
        column.lower(): column
        for column in schema_columns
    }

    # --------------------------------------------------------
    # Missing local columns
    # --------------------------------------------------------

    missing_columns = [
        schema_column
        for schema_column
        in schema_columns
        if schema_column.lower()
        not in local_lookup
    ]

    if missing_columns:

        raise RuntimeError(
            "Current local features are missing "
            "columns required by online Feature Group v2:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Extra local columns
    # --------------------------------------------------------

    extra_columns = [
        local_column
        for (
            lower_name,
            local_column,
        ) in local_lookup.items()
        if lower_name
        not in schema_lookup
    ]

    if extra_columns:

        print(
            "\nExtra local columns not in v2 schema:"
        )

        for column in extra_columns:

            print(
                f"  - {column}"
            )

        print(
            "These columns will be excluded."
        )

    # --------------------------------------------------------
    # Correct casing
    # --------------------------------------------------------

    rename_map = {}

    for schema_column in schema_columns:

        local_column = (
            local_lookup[
                schema_column.lower()
            ]
        )

        if local_column != schema_column:

            rename_map[
                local_column
            ] = schema_column

    if rename_map:

        dataframe = dataframe.rename(
            columns=rename_map
        )

    dataframe = dataframe[
        schema_columns
    ].copy()

    print(
        "\nSchema alignment successful."
    )

    return dataframe


# ============================================================
# Display Feature Vectors
# ============================================================

def display_latest_features(
    dataframe,
):
    """
    Show online rows before upload.
    """

    print("\n" + "=" * 70)
    print("LATEST ONLINE FEATURE VECTORS")
    print("=" * 70)

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
        if column in dataframe.columns
    ]

    print(
        "\n"
        + dataframe[
            columns
        ]
        .sort_values(
            "city"
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# Insert Into Online Store
# ============================================================

def insert_online_features(
    dataframe,
):
    """
    Write latest AQI features to Hopsworks Online Store only.
    """

    if dataframe is None:

        raise ValueError(
            "Online feature DataFrame is None."
        )

    if dataframe.empty:

        print(
            "\nNo feature rows are available "
            "for online ingestion."
        )

        return False

    print("\n" + "=" * 70)
    print("HOPSWORKS ONLINE FEATURE INGESTION")
    print("=" * 70)

    print(
        f"\nRows   : {len(dataframe)}"
    )

    print(
        f"Cities : "
        f"{dataframe['city'].nunique()}"
    )

    # ========================================================
    # Connect
    # ========================================================

    feature_store = (
        get_feature_store()
    )

    # ========================================================
    # Valid FeatureGroup object
    # ========================================================

    feature_group = (
        get_online_feature_group(
            feature_store
        )
    )

    if feature_group is None:

        raise RuntimeError(
            "Could not obtain a valid "
            "FeatureGroup object."
        )

    print(
        f"\nFeature Group object confirmed: "
        f"{type(feature_group).__name__}"
    )

    # ========================================================
    # Schema
    # ========================================================

    dataframe = (
        align_with_existing_schema(
            dataframe,
            feature_group,
        )
    )

    # ========================================================
    # Online-only ingestion
    # ========================================================

    print(
        "\nStarting ONLINE-ONLY upsert..."
    )

    print(
        "Offline HDFS materialization: DISABLED"
    )

    print(
        "Online Feature Store: ENABLED"
    )

    try:

        feature_group.insert(
            dataframe,

            operation="upsert",

            # Avoid the Windows -> HDFS offline path.
            storage="online",

            write_options={
                # External client.
                "internal_kafka":
                    False,

                "wait_for_online_ingestion":
                    True,

                "online_ingestion_options": {
                    "timeout":
                        120,

                    "period":
                        2,

                    # Do not replace a newer online value
                    # with an older event-time observation.
                    "upsert_if_newer":
                        True,

                    "mark_online_rows":
                        True,
                },
            },

            # Current HSFS versions use this to wait for
            # ingestion completion as well.
            wait=True,
        )

    except Exception as error:

        error_text = str(
            error
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "ONLINE FEATURE INGESTION FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"\nError:\n"
            f"{error_text}"
        )

        lower_error = (
            error_text.lower()
        )

        if (
            "hdfs"
            in lower_error
            or
            "gssapi"
            in lower_error
            or
            "delta"
            in lower_error
        ):

            print(
                "\nUnexpected offline-storage access "
                "was attempted."
            )

            print(
                "The insert call requested "
                "storage='online'."
            )

            print(
                "Check the installed Hopsworks/HSFS "
                "client version."
            )

        elif (
            "kafka"
            in lower_error
            or
            "broker"
            in lower_error
        ):

            print(
                "\nExternal online-ingestion/Kafka "
                "connectivity issue detected."
            )

        elif (
            "schema"
            in lower_error
            or
            "not compatible"
            in lower_error
        ):

            print(
                "\nFeature schema mismatch detected."
            )

        raise RuntimeError(
            "Hopsworks online feature ingestion failed.\n"
            f"Original error: {error_text}"
        ) from error

    # ========================================================
    # Optional ingestion status
    # ========================================================

    try:

        online_ingestion = (
            feature_group
            .get_latest_online_ingestion()
        )

        if online_ingestion is not None:

            print(
                "\nLatest online ingestion status "
                "retrieved successfully."
            )

            try:

                results = [
                    result.to_dict()
                    for result
                    in online_ingestion.results
                ]

                if results:

                    print(
                        f"Ingestion results: "
                        f"{results}"
                    )

            except Exception:

                pass

    except Exception:

        # Ingestion succeeded; inability to query its
        # observability object is not fatal.
        pass

    print("\n" + "=" * 70)
    print("ONLINE FEATURE INGESTION SUCCESSFUL")
    print("=" * 70)

    print(
        f"\nRows inserted : {len(dataframe)}"
    )

    print(
        f"Feature Group : "
        f"{REALTIME_FEATURE_GROUP_NAME}"
    )

    print(
        f"Version       : "
        f"{REALTIME_FEATURE_GROUP_VERSION}"
    )

    print(
        "Storage       : ONLINE"
    )

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI HOPSWORKS ONLINE FEATURE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 1: LOAD REAL-TIME FEATURES")
    print("-" * 70)

    dataframe = (
        load_realtime_features()
    )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 2: SELECT LATEST CITY FEATURES")
    print("-" * 70)

    dataframe = (
        select_latest_city_features(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 3: PREPARE ONLINE FEATURES")
    print("-" * 70)

    dataframe = (
        prepare_online_dataframe(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 4: DISPLAY ONLINE FEATURES")
    print("-" * 70)

    display_latest_features(
        dataframe
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 5: HOPSWORKS ONLINE INGESTION")
    print("-" * 70)

    insert_online_features(
        dataframe
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ONLINE FEATURE PIPELINE COMPLETED")
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()