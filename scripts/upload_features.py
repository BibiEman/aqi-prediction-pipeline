"""
AQI Feature Store Upload to Hopsworks

Uploads:
    data/processed/model_training_dataset.csv

Feature Group:
    aqi_features
    Version: 1

Designed for:
    Windows
    Hopsworks 5.x
    Python 3.11
"""

import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import hopsworks


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_training_dataset.csv"
)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Smaller batches reduce the chance of Windows/HDFS RPC failures.
BATCH_SIZE = 2000

# Local directories used by Hopsworks on Windows.
CERT_DIR = PROJECT_ROOT / ".hopsworks_certs"
TEMP_DIR = PROJECT_ROOT / ".hopsworks_tmp"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "city",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
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
    "target_aqi",
]


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# WINDOWS CONFIGURATION
# ============================================================

def configure_windows_environment():
    """
    Configure local directories so Hopsworks does not try to use
    the Unix-style /tmp path on Windows.
    """

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Hopsworks uses this environment variable for certificates.
    os.environ["HOPSWORKS_CERTS_DIR"] = str(CERT_DIR)

    # General temporary directories.
    os.environ["TMPDIR"] = str(TEMP_DIR)
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)

    print("Certificate folder:")
    print(CERT_DIR)

    print("\nTemporary directory:")
    print(TEMP_DIR)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    print_header("Loading dataset")

    print(f"Path: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_dataset(df):
    print_header("Validating dataset")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing_columns)
        )

    print("Required columns: OK")

    missing_values = int(
        df[REQUIRED_COLUMNS].isna().sum().sum()
    )

    print(f"Missing values: {missing_values}")

    if missing_values > 0:
        print("\nColumns containing missing values:")

        null_counts = (
            df[REQUIRED_COLUMNS]
            .isna()
            .sum()
        )

        print(
            null_counts[null_counts > 0]
            .to_string()
        )

        raise ValueError(
            "Dataset contains missing values."
        )

    print(f"Rows available: {len(df):,}")

    if len(df) == 0:
        raise ValueError("Dataset contains zero rows.")


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataset(df):
    print_header("Preparing dataset")

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    print("Converting timestamp...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found."
        )

    # Remove timezone information if present.
    try:
        if getattr(
            df["timestamp"].dt,
            "tz",
            None
        ) is not None:
            df["timestamp"] = (
                df["timestamp"]
                .dt
                .tz_localize(None)
            )
    except Exception:
        pass

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    print("Preparing categorical columns...")

    categorical_columns = [
        "city",
        "day_of_week",
        "season",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype(str)
        )

    # --------------------------------------------------------
    # Integer columns
    # --------------------------------------------------------

    print("Preparing integer columns...")

    integer_columns = [
        "humidity",
        "cloud_cover",
        "wind_direction",
        "us_aqi",
        "hour",
        "day",
        "month",
        "year",
        "is_weekend",
    ]

    for column in integer_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Invalid integer values in column: {column}"
            )

        df[column] = df[column].astype("int64")

    # --------------------------------------------------------
    # Floating point columns
    # --------------------------------------------------------

    print("Preparing floating-point columns...")

    float_columns = [
        "latitude",
        "longitude",
        "temperature",
        "pressure",
        "wind_speed",
        "precipitation",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
        "target_aqi",
    ]

    for column in float_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Invalid floating-point values "
                f"in column: {column}"
            )

        df[column] = df[column].astype("float64")

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    df = df[REQUIRED_COLUMNS]

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df.isna().sum().sum() != 0:
        raise ValueError(
            "Missing values appeared during preparation."
        )

    print("Dataset preparation: OK")
    print(f"Final dataset shape: {df.shape}")

    print("\nFinal columns:")

    for column in df.columns:
        print(f"  - {column}")

    return df


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

def connect_to_hopsworks():
    print_header("Connecting to Hopsworks")

    api_key = os.environ.get(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable "
            "is not set."
        )

    print("HOPSWORKS_API_KEY detected.")

    configure_windows_environment()

    print("\nLogging in to Hopsworks...")

    project = hopsworks.login(
        api_key_value=api_key
    )

    print("\nHopsworks login successful.")
    print(f"Project: {project.name}")

    return project


# ============================================================
# FEATURE STORE
# ============================================================

def get_feature_store(project):
    print_header("Accessing Feature Store")

    fs = project.get_feature_store()

    print("Feature Store connected successfully.")

    return fs


# ============================================================
# CREATE / GET FEATURE GROUP
# ============================================================

def create_feature_group(fs):
    print_header("Creating Feature Group")

    print(f"Name    : {FEATURE_GROUP_NAME}")
    print(f"Version : {FEATURE_GROUP_VERSION}")

    print("\nCreating/getting Feature Group...")

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "AQI prediction features including "
            "air quality, weather, temporal features, "
            "lag features, rolling features and target AQI."
        ),
        primary_key=[
            "city",
            "timestamp",
        ],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="DELTA",
    )

    print(
        "Feature Group created/found successfully."
    )

    return feature_group


# ============================================================
# UPLOAD DATA IN BATCHES
# ============================================================

def upload_data(
    feature_group,
    df,
    batch_size=BATCH_SIZE,
):
    print_header("Uploading Data")

    total_rows = len(df)

    print(f"Rows to upload: {total_rows:,}")
    print(f"Batch size    : {batch_size:,}")

    total_batches = (
        total_rows + batch_size - 1
    ) // batch_size

    uploaded = 0

    print("\nUploading to Hopsworks...")

    for start in range(
        0,
        total_rows,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total_rows,
        )

        batch = df.iloc[
            start:end
        ].copy()

        batch_number = (
            start // batch_size
        ) + 1

        print("\n" + "-" * 60)

        print(
            f"Batch {batch_number}/{total_batches}"
        )

        print(
            f"Rows: {start + 1:,} - {end:,}"
        )

        print(
            f"Batch size: {len(batch):,}"
        )

        print("-" * 60)

        try:

            feature_group.insert(
                batch,
                write_options={
                    "wait_for_job": True
                },
            )

            uploaded += len(batch)

            percentage = (
                uploaded / total_rows
            ) * 100

            print(
                f"Batch {batch_number} uploaded successfully."
            )

            print(
                f"Progress: "
                f"{uploaded:,}/{total_rows:,} "
                f"({percentage:.1f}%)"
            )

        except Exception as error:

            print_header(
                "BATCH UPLOAD FAILED"
            )

            print(
                f"Batch: "
                f"{batch_number}/{total_batches}"
            )

            print(
                f"Rows: "
                f"{start + 1:,}-{end:,}"
            )

            print(
                f"Successfully uploaded before failure: "
                f"{uploaded:,}"
            )

            print(
                f"\nError: {error}"
            )

            raise

    print_header(
        "ALL DATA UPLOADED SUCCESSFULLY"
    )

    print(
        f"Total rows uploaded: "
        f"{uploaded:,}"
    )

    return uploaded


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "AQI FEATURE STORE UPLOAD"
    )

    try:

        # ----------------------------------------------------
        # 1. Load
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # 2. Validate
        # ----------------------------------------------------

        validate_dataset(df)

        # ----------------------------------------------------
        # 3. Prepare
        # ----------------------------------------------------

        df = prepare_dataset(df)

        # ----------------------------------------------------
        # 4. Hopsworks
        # ----------------------------------------------------

        project = connect_to_hopsworks()

        # ----------------------------------------------------
        # 5. Feature Store
        # ----------------------------------------------------

        fs = get_feature_store(project)

        # ----------------------------------------------------
        # 6. Feature Group
        # ----------------------------------------------------

        feature_group = create_feature_group(fs)

        # ----------------------------------------------------
        # 7. Upload
        # ----------------------------------------------------

        uploaded = upload_data(
            feature_group,
            df,
            batch_size=BATCH_SIZE,
        )

        # ----------------------------------------------------
        # 8. Success
        # ----------------------------------------------------

        print_header(
            "FEATURE STORE UPLOAD COMPLETE"
        )

        print(
            f"Feature Group : "
            f"{FEATURE_GROUP_NAME}"
        )

        print(
            f"Version       : "
            f"{FEATURE_GROUP_VERSION}"
        )

        print(
            f"Rows uploaded : "
            f"{uploaded:,}"
        )

        print(
            "\nHopsworks Feature Store step "
            "completed successfully."
        )

    except Exception as error:

        print_header(
            "UPLOAD FAILED"
        )

        print(
            f"Error: {error}"
        )

        print("\nFull traceback:")

        traceback.print_exc()

        sys.exit(1)

    finally:

        # Hopsworks normally handles connection cleanup.
        pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()