"""
prepare_training.py

Prepare the final dataset for AQI forecasting.

Target:
    AQI three hours into the future.
"""

from pathlib import Path

import pandas as pd


# =====================================================
# Configuration
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_training_dataset.csv"
)

FORECAST_HORIZON = 3


# =====================================================
# Load Dataset
# =====================================================

def load_dataset():

    print("\nLoading feature-engineered dataset...")

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

    print(
        "Feature engineered dataset loaded successfully."
    )

    print(f"Shape: {df.shape}")

    return df


# =====================================================
# Create Target
# =====================================================

def create_target(df):
    """
    Create a three-hour-ahead AQI target.

    For each city:

        target_aqi(t)
            =
        us_aqi(t + 3 hours)
    """

    print(
        f"\nCreating {FORECAST_HORIZON}-hour "
        "forecast target..."
    )

    df = (
        df.sort_values(
            ["city", "timestamp"]
        )
        .reset_index(drop=True)
    )

    df["target_aqi"] = (
        df.groupby("city")["us_aqi"]
        .shift(-FORECAST_HORIZON)
    )

    return df


# =====================================================
# Validate Target
# =====================================================

def validate_target(df):

    print("\nTarget Validation")
    print("-" * 50)

    missing_target = (
        df["target_aqi"]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: {missing_target}"
    )

    if missing_target == len(df):
        raise ValueError(
            "All target values are missing."
        )

    print(
        f"Target range: "
        f"{df['target_aqi'].min():.2f} - "
        f"{df['target_aqi'].max():.2f}"
    )

    return df


# =====================================================
# Remove Incomplete Rows
# =====================================================

def remove_missing(df):

    print("\nRemoving incomplete rows...")

    before = len(df)

    df = df.dropna().reset_index(
        drop=True
    )

    after = len(df)

    print(
        f"Rows removed: {before - after}"
    )

    print(
        f"Remaining rows: {after}"
    )

    return df


# =====================================================
# Final Validation
# =====================================================

def validate_training_dataset(df):

    required_columns = [
        "timestamp",
        "city",
        "us_aqi",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
        "target_aqi",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required training columns: "
            f"{missing_columns}"
        )

    if df["target_aqi"].isna().any():

        raise ValueError(
            "Training dataset still contains "
            "missing target values."
        )

    return df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nModel training dataset saved successfully!"
    )

    print(OUTPUT_FILE)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 70)
    print("PREPARING AQI TRAINING DATASET")
    print("=" * 70)

    df = load_dataset()

    df = create_target(df)

    df = validate_target(df)

    df = remove_missing(df)

    df = validate_training_dataset(df)

    print(
        f"\nFinal Dataset Shape: {df.shape}"
    )

    print("\nFirst Five Rows:")
    print(df.head())

    save_dataset(df)


if __name__ == "__main__":
    main()