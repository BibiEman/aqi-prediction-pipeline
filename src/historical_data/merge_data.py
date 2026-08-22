"""
merge_data.py

Merge historical weather and air quality datasets
into one training dataset.
"""

import pandas as pd
from pathlib import Path


# =====================================================
# Load Datasets
# =====================================================

def load_datasets():

    project_root = Path(__file__).resolve().parents[2]

    weather_path = (
        project_root
        / "data"
        / "historical"
        / "weather"
        / "historical_weather.csv"
    )

    air_quality_path = (
        project_root
        / "data"
        / "historical"
        / "air_quality"
        / "historical_air_quality.csv"
    )

    weather_df = pd.read_csv(weather_path)

    air_quality_df = pd.read_csv(air_quality_path)

    return weather_df, air_quality_df


# =====================================================
# Merge
# =====================================================

def merge_datasets(weather_df, air_quality_df):

    merged_df = pd.merge(
        weather_df,
        air_quality_df,
        on=[
            "timestamp",
            "city",
            "latitude",
            "longitude"
        ],
        how="inner"
    )

    return merged_df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):

    project_root = Path(__file__).resolve().parents[2]

    output_path = (
        project_root
        / "data"
        / "historical"
        / "training_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nTraining dataset saved successfully!")
    print(output_path)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("Merging Historical Datasets")
    print("=" * 60)

    weather_df, air_quality_df = load_datasets()

    print("\nWeather Shape:", weather_df.shape)
    print("Air Quality Shape:", air_quality_df.shape)

    merged_df = merge_datasets(
        weather_df,
        air_quality_df
    )

    print("\nMerged Dataset")
    print(merged_df.head())

    print("\nMerged Shape:", merged_df.shape)

    save_dataset(merged_df)


if __name__ == "__main__":
    main()