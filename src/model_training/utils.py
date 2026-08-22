"""
utils.py

Utility functions for the AQI Prediction Pipeline.

Responsibilities:
    - Load model-training dataset
    - Separate features and target
    - Perform chronological train/test split
    - Build preprocessing pipeline
    - Save/load trained models

Important:
    Numerical features use an explicit FunctionTransformer
    instead of the string "passthrough".

    This avoids compatibility problems when a pipeline trained
    with one scikit-learn version is loaded by another version.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer,
)

from src.model_training.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
)


# =====================================================
# Load Dataset
# =====================================================

def load_dataset():
    """
    Load the processed model-training dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset sorted chronologically.
    """

    print("\nLoading dataset...")

    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    # -------------------------------------------------
    # Validate required columns
    # -------------------------------------------------

    required_columns = [
        "timestamp",
        "city",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            f"{missing_columns}"
        )

    # -------------------------------------------------
    # Convert timestamp
    # -------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found."
        )

    # -------------------------------------------------
    # Remove duplicate observations
    # -------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["timestamp", "city"]
    ).sum()

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count} "
            "duplicate city/timestamp rows..."
        )

        df = df.drop_duplicates(
            subset=["timestamp", "city"]
        )

    # -------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------

    df = (
        df.sort_values(
            ["timestamp", "city"]
        )
        .reset_index(drop=True)
    )

    print("Dataset loaded successfully!")

    print(
        f"Dataset Shape: {df.shape}"
    )

    print(
        f"Date Range: "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Cities: {df['city'].nunique()}"
    )

    return df


# =====================================================
# Split Features & Target
# =====================================================

def split_features_target(df):
    """
    Separate input features and target.

    Timestamp is removed from X because it is used
    only for chronological splitting.

    Returns
    -------
    X : pandas.DataFrame
        Input features.

    y : pandas.Series
        Target AQI.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    # -------------------------------------------------
    # Columns to remove
    # -------------------------------------------------

    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]

    X = df.drop(
        columns=columns_to_drop
    ).copy()

    y = df[TARGET_COLUMN].copy()

    print(
        f"\nFeatures Shape: {X.shape}"
    )

    print(
        f"Target Shape: {y.shape}"
    )

    return X, y


# =====================================================
# Chronological Train/Test Split
# =====================================================

def split_dataset(X, y):
    """
    Perform a chronological train/test split.

    The oldest observations are used for training
    and the newest observations are used for testing.

    This is appropriate for AQI forecasting because
    future observations must never be used to train
    the model.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y have different lengths."
        )

    if len(X) < 10:
        raise ValueError(
            "Dataset is too small for train/test split."
        )

    # -------------------------------------------------
    # Calculate split point
    # -------------------------------------------------

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    if split_index <= 0:
        raise ValueError(
            "Training dataset would be empty."
        )

    if split_index >= len(X):
        raise ValueError(
            "Testing dataset would be empty."
        )

    # -------------------------------------------------
    # Chronological split
    # -------------------------------------------------

    X_train = (
        X.iloc[:split_index]
        .copy()
    )

    X_test = (
        X.iloc[split_index:]
        .copy()
    )

    y_train = (
        y.iloc[:split_index]
        .copy()
    )

    y_test = (
        y.iloc[split_index:]
        .copy()
    )

    # -------------------------------------------------
    # Print information
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL TRAIN/TEST SPLIT")
    print("=" * 60)

    print(
        f"Training Samples : {len(X_train)}"
    )

    print(
        f"Testing Samples  : {len(X_test)}"
    )

    print(
        f"Training Ratio   : "
        f"{len(X_train) / len(X):.2%}"
    )

    print(
        f"Testing Ratio    : "
        f"{len(X_test) / len(X):.2%}"
    )

    # -------------------------------------------------
    # Validate ordering
    # -------------------------------------------------

    train_last_index = X_train.index.max()
    test_first_index = X_test.index.min()

    if train_last_index >= test_first_index:
        raise RuntimeError(
            "Temporal split validation failed."
        )

    print(
        "\nTemporal split completed successfully."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =====================================================
# Explicit Numerical Passthrough Transformer
# =====================================================

def numerical_passthrough(X):
    """
    Return numerical features unchanged.

    This function is intentionally explicit rather
    than using the string "passthrough".

    Using FunctionTransformer prevents the serialized
    ColumnTransformer from depending on sklearn's
    internal handling of the "passthrough" string.
    """

    return X


# =====================================================
# Build Preprocessor
# =====================================================

def build_preprocessor(X):
    """
    Build preprocessing transformer.

    Categorical:
        One-hot encoded.

    Numerical:
        Passed through unchanged using an explicit
        FunctionTransformer.

    Unknown categories during inference are ignored.
    """

    # -------------------------------------------------
    # Identify categorical features
    # -------------------------------------------------

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in X.columns
    ]

    # -------------------------------------------------
    # Identify numerical features
    # -------------------------------------------------

    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    transformers = []

    # -------------------------------------------------
    # Categorical preprocessing
    # -------------------------------------------------

    if categorical_features:

        transformers.append(
            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),

                categorical_features,
            )
        )

    # -------------------------------------------------
    # Numerical preprocessing
    #
    # IMPORTANT:
    # Do NOT use:
    #
    #     "passthrough"
    #
    # Use an actual transformer object instead.
    # -------------------------------------------------

    if numerical_features:

        numerical_transformer = (
            FunctionTransformer(
                numerical_passthrough,
                validate=False,
            )
        )

        transformers.append(
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            )
        )

    # -------------------------------------------------
    # Validate transformers
    # -------------------------------------------------

    if not transformers:
        raise ValueError(
            "No valid preprocessing features were found."
        )

    # -------------------------------------------------
    # Create ColumnTransformer
    # -------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# =====================================================
# Build Complete Pipeline
# =====================================================

def build_pipeline(model, X):
    """
    Create complete preprocessing + model pipeline.

    The pipeline contains:

        raw DataFrame
            ↓
        ColumnTransformer
            ↓
        OneHotEncoder / FunctionTransformer
            ↓
        ML model

    The exact preprocessing used during training
    is therefore automatically reused during prediction.
    """

    preprocessor = build_preprocessor(
        X
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# =====================================================
# Save Model
# =====================================================

def save_model(model, filepath):
    """
    Save a trained model/pipeline using joblib.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        filepath,
    )

    print("\nModel saved:")
    print(filepath)


# =====================================================
# Load Model
# =====================================================

def load_model(filepath):
    """
    Load a trained model/pipeline.
    """

    filepath = Path(filepath)

    if not filepath.exists():

        raise FileNotFoundError(
            f"Model not found:\n{filepath}"
        )

    model = joblib.load(
        filepath
    )

    print(
        "\nModel loaded successfully."
    )

    return model