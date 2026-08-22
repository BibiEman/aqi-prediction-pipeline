"""
build_features.py

Create machine learning features from the cleaned AQI dataset.
"""

from pathlib import Path

import pandas as pd


# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_training_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_dataset.csv"
)


# =====================================================
# Load Dataset
# =====================================================

def load_dataset():
    """
    Load the cleaned dataset.
    """

    print("\nLoading cleaned dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found."
        )

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


# =====================================================
# Validate Dataset
# =====================================================

def validate_dataset(df):
    """
    Validate required columns.
    """

    required_columns = [
        "timestamp",
        "city",
        "us_aqi",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# =====================================================
# Sort Dataset
# =====================================================

def sort_dataset(df):
    """
    Sort observations by city and timestamp.
    """

    print("\nSorting dataset...")

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    return df


# =====================================================
# Create Time Features
# =====================================================

def create_time_features(df):
    """
    Extract calendar/time features.
    """

    print("Creating time features...")

    df["hour"] = df["timestamp"].dt.hour

    df["day"] = df["timestamp"].dt.day

    df["month"] = df["timestamp"].dt.month

    df["year"] = df["timestamp"].dt.year

    df["day_of_week"] = (
        df["timestamp"]
        .dt.day_name()
    )

    df["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        if month in [3, 4, 5]:
            return "Spring"

        if month in [6, 7, 8]:
            return "Summer"

        return "Autumn"

    df["season"] = (
        df["month"]
        .apply(get_season)
    )

    return df


# =====================================================
# Create Lag Features
# =====================================================

def create_lag_features(df):
    """
    Create historical AQI lag features separately
    for every city.
    """

    print("Creating lag features...")

    grouped_aqi = (
        df.groupby("city")["us_aqi"]
    )

    df["aqi_lag_1"] = (
        grouped_aqi.shift(1)
    )

    df["aqi_lag_3"] = (
        grouped_aqi.shift(3)
    )

    df["aqi_lag_6"] = (
        grouped_aqi.shift(6)
    )

    df["aqi_lag_24"] = (
        grouped_aqi.shift(24)
    )

    return df


# =====================================================
# Create Rolling Features
# =====================================================

def create_rolling_features(df):
    """
    Create historical rolling AQI averages.

    IMPORTANT:
    The rolling windows use only current/past
    observations. They never use future AQI.
    """

    print("Creating rolling features...")

    grouped_aqi = (
        df.groupby("city")["us_aqi"]
    )

    df["aqi_roll_3"] = (
        grouped_aqi
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=3
            ).mean()
        )
    )

    df["aqi_roll_6"] = (
        grouped_aqi
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=6
            ).mean()
        )
    )

    df["aqi_roll_24"] = (
        grouped_aqi
        .transform(
            lambda x: x.rolling(
                window=24,
                min_periods=24
            ).mean()
        )
    )

    return df


# =====================================================
# Feature Validation
# =====================================================

def validate_features(df):
    """
    Validate generated features.
    """

    feature_columns = [
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
    ]

    print("\nFeature Validation")
    print("-" * 50)

    for column in feature_columns:

        missing = df[column].isna().sum()

        print(
            f"{column:15s}: "
            f"{missing} missing values"
        )

    return df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):
    """
    Save feature-engineered dataset.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nFeature engineered dataset saved successfully!"
    )

    print(OUTPUT_FILE)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 70)
    print("AQI FEATURE ENGINEERING")
    print("=" * 70)

    df = load_dataset()

    df = validate_dataset(df)

    df = sort_dataset(df)

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = validate_features(df)

    print("\nFeature Engineering Completed!")

    print(
        f"\nDataset Shape: {df.shape}"
    )

    print("\nFirst Five Rows:")
    print(df.head())

    save_dataset(df)


if __name__ == "__main__":
    main()