"""
preprocess_data.py

Clean the historical training dataset
before feature engineering.
"""

import pandas as pd
from pathlib import Path


# =====================================================
# Load Dataset
# =====================================================

def load_dataset():
    """
    Load the merged historical dataset.
    """

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "historical"
        / "training_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    print("Dataset loaded successfully.")

    return df


# =====================================================
# Display Dataset Information
# =====================================================

def inspect_dataset(df):
    """
    Display dataset overview.
    """

    print("\n" + "=" * 60)
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

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    print(df.duplicated().sum())


# =====================================================
# Clean Dataset
# =====================================================

def clean_dataset(df):
    """
    Clean the dataset.
    """

    print("\nConverting timestamp...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Sorting data...")

    df = df.sort_values(
        by=["city", "timestamp"]
    )

    print("Removing duplicates...")

    df = df.drop_duplicates()

    print("Cleaning completed.")

    return df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):
    """
    Save cleaned dataset.
    """

    project_root = Path(__file__).resolve().parents[2]

    output_folder = (
        project_root
        / "data"
        / "processed"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder
        / "cleaned_training_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nCleaned dataset saved successfully!")

    print(output_file)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("AQI DATA PREPROCESSING")
    print("=" * 60)

    df = load_dataset()

    inspect_dataset(df)

    df = clean_dataset(df)

    save_dataset(df)


if __name__ == "__main__":
    main()