
"""
train_models.py

AQI Model Training Pipeline

Models:
    - Ridge Regression
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost

The pipeline:
    1. Loads the AQI dataset.
    2. Uses the local Parquet cache when available.
    3. Falls back to Hopsworks if the cache is unavailable.
    4. Creates a chronological train/test split.
    5. Trains seven regression models.
    6. Saves complete preprocessing + model pipelines.
    7. Saves training metadata.
"""

from pathlib import Path
import time
import warnings

import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.model_training.config import (
    MODEL_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    ET_PARAMS,
    GB_PARAMS,
    XGB_PARAMS,
    LGBM_PARAMS,
    CAT_PARAMS,
)

from src.model_training.utils import (
    split_features_target,
    split_dataset,
    build_pipeline,
)

from src.model_training.hopsworks_dataset import (
    load_hopsworks_dataset,
)

warnings.filterwarnings("ignore")


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

CACHE_FILE = CACHE_DIR / "aqi_features.parquet"

MODEL_DIR = Path(MODEL_DIR)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Dataset Loading
# ============================================================

def load_training_data():
    """
    Load training data.

    Priority:
        1. Local Parquet cache
        2. Hopsworks Feature Store

    Returns
    -------
    pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("AQI TRAINING DATA LOADER")
    print("=" * 70)

    # --------------------------------------------------------
    # Option 1: Local cache
    # --------------------------------------------------------

    if CACHE_FILE.exists():

        print("\nLocal dataset cache found.")

        print(
            f"Loading:\n"
            f"{CACHE_FILE}"
        )

        try:

            df = pd.read_parquet(
                CACHE_FILE
            )

            print(
                "\nDataset loaded from local cache."
            )

            return validate_dataset(
                df
            )

        except Exception as error:

            print(
                "\nWARNING: Local cache could not be read."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "\nTrying Hopsworks..."
            )

    # --------------------------------------------------------
    # Option 2: Hopsworks
    # --------------------------------------------------------

    print(
        "\nLocal cache unavailable."
    )

    print(
        "Loading dataset from Hopsworks..."
    )

    df = load_hopsworks_dataset()

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    try:

        df.to_parquet(
            CACHE_FILE,
            index=False,
        )

        print(
            "\nHopsworks dataset cached locally:"
        )

        print(
            CACHE_FILE
        )

    except Exception as error:

        print(
            "\nWARNING: Could not save local cache."
        )

        print(
            f"Reason: {error}"
        )

    return validate_dataset(
        df
    )


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(df):
    """
    Validate the training DataFrame.
    """

    if df is None:

        raise ValueError(
            "Dataset is None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Training data must be a pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    required_columns = [
        "timestamp",
        "city",
        "target_aqi",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = (
        df["timestamp"]
        .isna()
        .sum()
    )

    if invalid_timestamp_count > 0:

        raise ValueError(
            "Invalid timestamp values found: "
            f"{invalid_timestamp_count}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df["target_aqi"] = pd.to_numeric(
        df["target_aqi"],
        errors="coerce",
    )

    missing_target_count = (
        df["target_aqi"]
        .isna()
        .sum()
    )

    if missing_target_count > 0:

        raise ValueError(
            "Missing target_aqi values: "
            f"{missing_target_count}"
        )

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    missing_city_count = (
        df["city"]
        .isna()
        .sum()
    )

    if missing_city_count > 0:

        raise ValueError(
            "Missing city values: "
            f"{missing_city_count}"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET READY"
    )

    print(
        "=" * 70
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
        f"Cities     : "
        f"{df['city'].nunique()}"
    )

    print(
        f"Target     : target_aqi"
    )

    return df


# ============================================================
# Model Definitions
# ============================================================

def create_models():
    """
    Create all AQI regression models.
    """

    models = {

        "Ridge": Ridge(
            alpha=1.0,
        ),

        "RandomForest": RandomForestRegressor(
            **RF_PARAMS
        ),

        "ExtraTrees": ExtraTreesRegressor(
            **ET_PARAMS
        ),

        "GradientBoosting": GradientBoostingRegressor(
            **GB_PARAMS
        ),

        "XGBoost": XGBRegressor(
            **XGB_PARAMS
        ),

        "LightGBM": LGBMRegressor(
            **LGBM_PARAMS
        ),

        "CatBoost": CatBoostRegressor(
            **CAT_PARAMS
        ),
    }

    return models


# ============================================================
# Train One Model
# ============================================================

def train_single_model(
    model_name,
    model,
    X_train,
    y_train,
):
    """
    Train and save one complete pipeline.

    The saved object contains:
        preprocessing
        +
        trained model
    """

    print("\n" + "=" * 70)
    print(
        f"TRAINING MODEL: {model_name}"
    )
    print("=" * 70)

    print(
        f"Training rows : {len(X_train):,}"
    )

    print(
        f"Features      : {X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    print(
        "\nBuilding preprocessing pipeline..."
    )

    pipeline = build_pipeline(
        model,
        X_train,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Starting model training..."
    )

    start_time = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    training_time = round(
        elapsed_time,
        2,
    )

    print(
        f"\n{model_name} training completed."
    )

    print(
        f"Training time: "
        f"{training_time} seconds"
    )

    # --------------------------------------------------------
    # Save with joblib
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    print(
        "\nSaving model..."
    )

    joblib.dump(
        pipeline,
        model_path,
        compress=3,
    )

    print(
        f"Model saved to:"
    )

    print(
        model_path
    )

    # --------------------------------------------------------
    # Verify saved model
    # --------------------------------------------------------

    print(
        "\nVerifying saved model..."
    )

    if not model_path.exists():

        raise RuntimeError(
            f"Model file was not created: "
            f"{model_path}"
        )

    file_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Saved file size: "
        f"{file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Test loading
    # --------------------------------------------------------

    try:

        loaded_pipeline = joblib.load(
            model_path
        )

        print(
            "Model reload test: PASSED"
        )

    except Exception as error:

        raise RuntimeError(
            f"Saved model could not be reloaded:\n"
            f"{error}"
        ) from error

    return {
        "Model": model_name,
        "Training Time (sec)": training_time,
        "Model Path": str(
            model_path
        ),
        "Status": "SUCCESS",
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "AQI MODEL TRAINING PIPELINE"
    )
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 1: LOADING TRAINING DATA"
    )
    print("-" * 70)

    df = load_training_data()

    # ========================================================
    # STEP 2
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 2: PREPARING FEATURES AND TARGET"
    )
    print("-" * 70)

    X, y = split_features_target(
        df
    )

    print(
        f"\nFeatures : {X.shape[1]}"
    )

    print(
        f"Target   : {y.name}"
    )

    print(
        f"Rows     : {len(X):,}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT"
    )
    print("-" * 70)

    X_train, X_test, y_train, y_test = (
        split_dataset(
            X,
            y,
        )
    )

    print(
        f"\nTraining rows : {len(X_train):,}"
    )

    print(
        f"Testing rows  : {len(X_test):,}"
    )

    print(
        f"Test size     : "
        f"{len(X_test) / len(X) * 100:.1f}%"
    )

    # ========================================================
    # Save evaluation split
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "SAVING EVALUATION DATA"
    )
    print("-" * 70)

    evaluation_data_path = (
        MODEL_DIR
        / "evaluation_test_data.joblib"
    )

    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
        },
        evaluation_data_path,
        compress=3,
    )

    print(
        f"Evaluation data saved to:"
    )

    print(
        evaluation_data_path
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 4: DEFINING MACHINE LEARNING MODELS"
    )
    print("-" * 70)

    models = create_models()

    print(
        f"\nTotal models: "
        f"{len(models)}"
    )

    for model_name in models:

        print(
            f"  - {model_name}"
        )

    # ========================================================
    # STEP 5
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 5: TRAINING ALL MODELS"
    )
    print("-" * 70)

    results = []

    failed_models = []

    for model_name, model in models.items():

        try:

            result = train_single_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                "\n" + "!" * 70
            )

            print(
                f"ERROR TRAINING {model_name}"
            )

            print(
                "!" * 70
            )

            print(
                f"Error: {error}"
            )

            failed_models.append(
                {
                    "Model": model_name,
                    "Error": str(error),
                }
            )

    # ========================================================
    # STEP 6
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 6: SAVING TRAINING SUMMARY"
    )
    print("-" * 70)

    if results:

        summary = pd.DataFrame(
            results
        )

        summary_path = (
            MODEL_DIR
            / "training_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print(
            "\nTraining Summary:"
        )

        print(
            summary.to_string(
                index=False
            )
        )

        print(
            f"\nTraining summary saved to:"
        )

        print(
            summary_path
        )

    else:

        raise RuntimeError(
            "No models were successfully trained."
        )

    # ========================================================
    # Failed models
    # ========================================================

    if failed_models:

        failed_summary = pd.DataFrame(
            failed_models
        )

        failed_path = (
            MODEL_DIR
            / "failed_models.csv"
        )

        failed_summary.to_csv(
            failed_path,
            index=False,
        )

        print(
            "\nWARNING:"
        )

        print(
            f"{len(failed_models)} "
            "model(s) failed."
        )

        print(
            failed_summary.to_string(
                index=False
            )
        )

        print(
            f"\nFailure details saved to:"
        )

        print(
            failed_path
        )

    # ========================================================
    # STEP 7
    # ========================================================

    print("\n" + "-" * 70)
    print(
        "STEP 7: SAVING TRAINING METADATA"
    )
    print("-" * 70)

    metadata = {

        "dataset_rows": len(df),

        "dataset_columns": len(df.columns),

        "feature_count": X.shape[1],

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "test_size": 0.20,

        "target": "target_aqi",

        "random_state": RANDOM_STATE,

        "models_requested": len(models),

        "models_successfully_trained": len(
            results
        ),

        "models_failed": len(
            failed_models
        ),

        "cache_file": str(
            CACHE_FILE
        ),

        "evaluation_data": str(
            evaluation_data_path
        ),
    }

    metadata_path = (
        MODEL_DIR
        / "training_metadata.joblib"
    )

    joblib.dump(
        metadata,
        metadata_path,
        compress=3,
    )

    print(
        f"Training metadata saved to:"
    )

    print(
        metadata_path
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "MODEL TRAINING COMPLETED"
    )
    print("=" * 70)

    print(
        f"\nModels requested      : "
        f"{len(models)}"
    )

    print(
        f"Models trained        : "
        f"{len(results)}"
    )

    print(
        f"Models failed         : "
        f"{len(failed_models)}"
    )

    print(
        "\nModel directory:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nGenerated files:"
    )

    for result in results:

        print(
            f"  ✓ {result['Model']}.joblib"
        )

    print(
        "  ✓ training_summary.csv"
    )

    print(
        "  ✓ evaluation_test_data.joblib"
    )

    print(
        "  ✓ training_metadata.joblib"
    )

    if failed_models:

        print(
            "\nWARNING:"
        )

        print(
            "Some models failed. "
            "Check failed_models.csv."
        )

    else:

        print(
            "\nALL 7 MODELS TRAINED SUCCESSFULLY."
        )


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()
