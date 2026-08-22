"""
preprocess_data.py

This module loads the historical AQI dataset,
performs basic preprocessing and saves
a cleaned dataset for feature engineering.
"""

from pathlib import Path
import pandas as pd


# =====================================================
# File Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "original"
    / "pakistan_air_quality_final_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "cleaned_historical_data.csv"


# =====================================================
# Load Dataset
# =====================================================

def load_data(file_path):
    """
    Load CSV file into a pandas DataFrame.
    """

    print("\nLoading dataset...")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.\n")

    return df


# =====================================================
# Dataset Information
# =====================================================

def explore_data(df):
    """
    Display basic information about the dataset.
    """

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst Five Rows:")
    print(df.head())


# =====================================================
# Missing Values
# =====================================================

def check_missing_values(df):
    """
    Display missing values.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    return missing


# =====================================================
# Duplicate Rows
# =====================================================

def check_duplicates(df):
    """
    Display duplicate rows.
    """

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows: {duplicates}")

    return duplicates


# =====================================================
# Timestamp Conversion
# =====================================================

def convert_timestamp(df):
    """
    Convert timestamp column to datetime.
    """

    print("\nConverting timestamp...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Timestamp converted successfully.")

    return df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):
    """
    Save cleaned dataset.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaned dataset saved successfully!")

    print(OUTPUT_FILE)


# =====================================================
# Main Function
# =====================================================

def main():

    df = load_data(INPUT_FILE)

    explore_data(df)

    check_missing_values(df)

    check_duplicates(df)

    df = convert_timestamp(df)

    save_dataset(df)


if __name__ == "__main__":
    main()