"""
hopsworks_dataset.py

Load the AQI training dataset from Hopsworks Feature Store.

The dataset is cached locally as Parquet so that model training
does not need to repeatedly query the Hopsworks Feature Query
Service.
"""

from pathlib import Path

import pandas as pd
import hopsworks


# ============================================================
# Hopsworks Configuration
# ============================================================

PROJECT_NAME = "AQI_Prediction_3days"

FEATURE_GROUP_NAME = "aqi_features"

FEATURE_GROUP_VERSION = 1


# ============================================================
# Local Cache
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

CACHE_FILE = CACHE_DIR / "aqi_features.parquet"


# ============================================================
# Required Columns
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "city",
    "target_aqi",
]


# ============================================================
# Validate Dataset
# ============================================================

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the AQI dataset.
    """

    if df is None:
        raise ValueError(
            "Dataset is None."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected pandas DataFrame, "
            f"received {type(df)}"
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            f"{missing_columns}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            "Invalid timestamp values found: "
            f"{invalid_timestamps}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    missing_target = df["target_aqi"].isna().sum()

    if missing_target > 0:
        raise ValueError(
            "Missing target_aqi values found: "
            f"{missing_target}"
        )

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    missing_city = df["city"].isna().sum()

    if missing_city > 0:
        raise ValueError(
            "Missing city values found: "
            f"{missing_city}"
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "timestamp",
            "city",
        ]
    ).sum()

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count:,} "
            "duplicate city/timestamp rows..."
        )

        df = df.drop_duplicates(
            subset=[
                "timestamp",
                "city",
            ],
            keep="last",
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
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Load From Hopsworks
# ============================================================

def load_from_hopsworks() -> pd.DataFrame:
    """
    Load the AQI dataset from Hopsworks.

    This function is called when the local cache does not exist.
    """

    print("\n" + "=" * 60)
    print("LOADING DATA FROM HOPSWORKS FEATURE STORE")
    print("=" * 60)

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        project=PROJECT_NAME
    )

    print(
        f"Project: {PROJECT_NAME}"
    )

    # --------------------------------------------------------
    # Feature Store
    # --------------------------------------------------------

    print(
        "\nAccessing Feature Store..."
    )

    fs = project.get_feature_store()

    # --------------------------------------------------------
    # Feature Group
    # --------------------------------------------------------

    print(
        "Accessing Feature Group..."
    )

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print(
        f"Feature Group: {FEATURE_GROUP_NAME}"
    )

    print(
        f"Version: {FEATURE_GROUP_VERSION}"
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    print(
        "\nReading Feature Group..."
    )

    df = fg.read(
        dataframe_type="pandas"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    df = validate_dataset(df)

    # --------------------------------------------------------
    # Save Cache
    # --------------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving local training cache..."
    )

    df.to_parquet(
        CACHE_FILE,
        index=False,
    )

    print(
        f"Cache saved to:\n{CACHE_FILE}"
    )

    return df


# ============================================================
# Load Dataset
# ============================================================

def load_hopsworks_dataset(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load the AQI training dataset.

    If a local Parquet cache exists, use it.
    Otherwise download the dataset from Hopsworks.

    Parameters
    ----------
    force_refresh : bool
        If True, ignore the local cache and download again.

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 60)
    print("AQI TRAINING DATA LOADER")
    print("=" * 60)

    # ========================================================
    # Use Local Cache
    # ========================================================

    if CACHE_FILE.exists() and not force_refresh:

        print(
            "\nLocal dataset cache found."
        )

        print(
            f"Loading:\n{CACHE_FILE}"
        )

        df = pd.read_parquet(
            CACHE_FILE
        )

        df = validate_dataset(df)

        print(
            "\nDataset loaded from local cache."
        )

    # ========================================================
    # Download From Hopsworks
    # ========================================================

    else:

        if force_refresh:

            print(
                "\nForce refresh requested."
            )

        else:

            print(
                "\nNo local cache found."
            )

        df = load_from_hopsworks()

    # ========================================================
    # Information
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"Date range : "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Cities     : {df['city'].nunique()}"
    )

    print(
        f"Target     : target_aqi"
    )

    return df


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    try:

        dataframe = load_hopsworks_dataset()

        print(
            "\n" + "=" * 60
        )

        print(
            "HOPSWORKS DATASET TEST SUCCESSFUL"
        )

        print(
            "=" * 60
        )

        print(
            f"\nShape: {dataframe.shape}"
        )

        print(
            "\nFirst 5 rows:"
        )

        print(
            dataframe.head()
        )

    except Exception as error:

        print(
            "\n" + "=" * 60
        )

        print(
            "HOPSWORKS DATASET TEST FAILED"
        )

        print(
            "=" * 60
        )

        print(
            f"\nError: {error}"
        )

        raise