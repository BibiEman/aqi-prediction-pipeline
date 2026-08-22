"""
collect_data.py

Real-time AQI + weather data collection pipeline.

This script:

1. Fetches current weather for all supported cities.
2. Fetches current air-pollution data.
3. Merges weather and pollution observations.
4. Appends new observations to the historical real-time dataset.
5. Removes duplicate city/timestamp observations.
6. Saves the cumulative dataset.

Output
------
data/raw/aqi_dataset.csv
"""

from pathlib import Path

import pandas as pd

from src.data_collection.weather_fetch import (
    collect_weather_data,
)

from src.data_collection.air_quality import (
    get_air_quality,
)


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REALTIME_DATASET_PATH = (
    RAW_DATA_DIR
    / "aqi_dataset.csv"
)


# ============================================================
# Supported Cities
# ============================================================

PAKISTAN_CITIES = [
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
# Merge Weather and AQI
# ============================================================

def merge_weather_and_aqi(
    weather_df,
):
    """
    Add air-pollution information to weather observations.

    Parameters
    ----------
    weather_df : pandas.DataFrame
        Current weather observations.

    Returns
    -------
    pandas.DataFrame
        Combined weather + air-quality observations.
    """

    if weather_df is None:

        raise ValueError(
            "Weather DataFrame is None."
        )

    if weather_df.empty:

        raise ValueError(
            "Weather DataFrame is empty."
        )

    required_columns = [
        "timestamp",
        "city",
        "latitude",
        "longitude",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weather_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Weather data is missing required columns:\n"
            f"{missing_columns}"
        )

    merged_records = []

    print(
        "\n" + "=" * 70
    )

    print(
        "COLLECTING AIR QUALITY DATA"
    )

    print(
        "=" * 70
    )

    for _, row in weather_df.iterrows():

        city = str(
            row["city"]
        )

        print(
            f"Fetching AQI for {city}..."
        )

        try:

            air_data = get_air_quality(
                latitude=float(
                    row["latitude"]
                ),
                longitude=float(
                    row["longitude"]
                ),
            )

        except Exception as error:

            print(
                f"WARNING: AQI request failed "
                f"for {city}: {error}"
            )

            continue

        if not air_data:

            print(
                f"WARNING: No AQI data returned "
                f"for {city}."
            )

            continue

        record = (
            row.to_dict()
        )

        record.update(
            air_data
        )

        merged_records.append(
            record
        )

    final_df = pd.DataFrame(
        merged_records
    )

    if final_df.empty:

        raise RuntimeError(
            "No merged weather/AQI records "
            "were created."
        )

    return final_df


# ============================================================
# Standardize Dataset
# ============================================================

def standardize_dataset(
    df,
):
    """
    Standardize timestamps, city names and numeric values.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" not in df.columns:

        raise ValueError(
            "Dataset does not contain timestamp."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = int(
        df["timestamp"]
        .isna()
        .sum()
    )

    if invalid_timestamps > 0:

        print(
            f"\nRemoving {invalid_timestamps} "
            "rows with invalid timestamps..."
        )

        df = df[
            df["timestamp"].notna()
        ].copy()

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    if "city" not in df.columns:

        raise ValueError(
            "Dataset does not contain city."
        )

    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# Load Existing Historical Data
# ============================================================

def load_existing_dataset():
    """
    Load previously collected real-time observations.

    Returns
    -------
    pandas.DataFrame or None
    """

    if not REALTIME_DATASET_PATH.exists():

        print(
            "\nNo existing real-time dataset found."
        )

        print(
            "A new dataset will be created."
        )

        return None

    try:

        existing_df = pd.read_csv(
            REALTIME_DATASET_PATH
        )

    except Exception as error:

        raise RuntimeError(
            "Could not read existing real-time dataset.\n"
            f"Path: {REALTIME_DATASET_PATH}\n"
            f"Error: {error}"
        ) from error

    if existing_df.empty:

        return None

    existing_df = standardize_dataset(
        existing_df
    )

    print(
        "\nExisting real-time dataset loaded."
    )

    print(
        f"Existing rows : "
        f"{len(existing_df):,}"
    )

    print(
        f"Date range    : "
        f"{existing_df['timestamp'].min()} "
        f"→ "
        f"{existing_df['timestamp'].max()}"
    )

    return existing_df


# ============================================================
# Combine Historical + New Data
# ============================================================

def combine_datasets(
    existing_df,
    new_df,
):
    """
    Append new observations to historical real-time data.
    """

    new_df = standardize_dataset(
        new_df
    )

    if existing_df is None:

        combined_df = (
            new_df.copy()
        )

    else:

        # ----------------------------------------------------
        # Align columns
        # ----------------------------------------------------

        all_columns = list(
            dict.fromkeys(
                list(existing_df.columns)
                + list(new_df.columns)
            )
        )

        existing_df = (
            existing_df.reindex(
                columns=all_columns
            )
        )

        new_df = (
            new_df.reindex(
                columns=all_columns
            )
        )

        combined_df = pd.concat(
            [
                existing_df,
                new_df,
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Duplicate Detection
    # --------------------------------------------------------

    before = len(
        combined_df
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=[
                "timestamp",
                "city",
            ],
            keep="last",
        )
    )

    after = len(
        combined_df
    )

    duplicates_removed = (
        before
        - after
    )

    if duplicates_removed > 0:

        print(
            f"\nRemoved "
            f"{duplicates_removed:,} "
            "duplicate observations."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    combined_df = (
        combined_df
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

    return combined_df


# ============================================================
# Save Dataset
# ============================================================

def save_dataset(
    df,
):
    """
    Save cumulative real-time dataset.
    """

    if df.empty:

        raise ValueError(
            "Cannot save an empty dataset."
        )

    REALTIME_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        REALTIME_DATASET_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL-TIME DATASET SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFile:\n"
        f"{REALTIME_DATASET_PATH}"
    )

    print(
        f"\nTotal rows : "
        f"{len(df):,}"
    )

    print(
        f"Cities     : "
        f"{df['city'].nunique()}"
    )

    print(
        f"Date range : "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )


# ============================================================
# Display Current Collection
# ============================================================

def display_collection_summary(
    new_df,
):
    """
    Display latest collection batch.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CURRENT COLLECTION SUMMARY"
    )

    print(
        "=" * 70
    )

    display_columns = [
        column
        for column in [
            "timestamp",
            "city",
            "temperature",
            "humidity",
            "aqi",
            "pm2_5",
            "pm10",
        ]
        if column in new_df.columns
    ]

    print(
        "\n"
        + new_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nNew observations : "
        f"{len(new_df)}"
    )


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "AQI REAL-TIME DATA COLLECTION PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCities configured: "
        f"{len(PAKISTAN_CITIES)}"
    )

    for city in PAKISTAN_CITIES:

        print(
            f"  - {city}"
        )

    # ========================================================
    # Step 1: Weather
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 1: COLLECTING CURRENT WEATHER"
    )

    print(
        "-" * 70
    )

    weather_df = collect_weather_data(
        PAKISTAN_CITIES
    )

    if weather_df is None:

        raise RuntimeError(
            "Weather collection returned None."
        )

    if weather_df.empty:

        raise RuntimeError(
            "Weather collection returned no data."
        )

    print(
        f"\nWeather observations collected: "
        f"{len(weather_df)}"
    )

    # ========================================================
    # Step 2: AQI
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 2: MERGING AIR QUALITY DATA"
    )

    print(
        "-" * 70
    )

    current_df = merge_weather_and_aqi(
        weather_df
    )

    current_df = standardize_dataset(
        current_df
    )

    display_collection_summary(
        current_df
    )

    # ========================================================
    # Step 3: Existing Dataset
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 3: LOADING EXISTING REAL-TIME HISTORY"
    )

    print(
        "-" * 70
    )

    existing_df = (
        load_existing_dataset()
    )

    # ========================================================
    # Step 4: Append
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 4: APPENDING NEW OBSERVATIONS"
    )

    print(
        "-" * 70
    )

    combined_df = combine_datasets(
        existing_df=existing_df,
        new_df=current_df,
    )

    # ========================================================
    # Step 5: Save
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 5: SAVING CUMULATIVE DATASET"
    )

    print(
        "-" * 70
    )

    save_dataset(
        combined_df
    )

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL-TIME COLLECTION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nNew observations collected : "
        f"{len(current_df)}"
    )

    print(
        f"Historical observations    : "
        f"{len(combined_df):,}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()