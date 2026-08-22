"""
compare_models.py

Evaluate all trained AQI prediction models,
select the best model, register it, and promote
it to production only when it outperforms the
current production version.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.model_training.config import (
    MODEL_DIR,
    RESULTS_DIR,
)

from src.model_training.utils import (
    load_dataset,
    split_features_target,
    split_dataset,
)

from src.model_training.model_registry import (
    register_and_promote_if_better,
)


# =====================================================
# Models
# =====================================================

MODEL_NAMES = [
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


# =====================================================
# MAPE
# =====================================================

def calculate_mape(
    y_true,
    y_pred
):
    """
    Calculate MAPE safely.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    mask = (
        y_true != 0
    )

    if not np.any(mask):

        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                /
                y_true[mask]
            )
        )
        * 100
    )


# =====================================================
# Dataset Version
# =====================================================

def get_dataset_version(df):
    """
    Generate dataset version from
    timestamp range.
    """

    if "timestamp" not in df.columns:

        return "unknown"

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    start_date = (
        timestamps.min()
        .strftime("%Y-%m-%d")
    )

    end_date = (
        timestamps.max()
        .strftime("%Y-%m-%d")
    )

    return (
        f"{start_date}_to_{end_date}"
    )


# =====================================================
# Evaluate One Model
# =====================================================

def evaluate_model(
    model_name,
    model_path,
    X_test,
    y_test,
):
    """
    Evaluate a single trained model.
    """

    print("\n" + "=" * 60)

    print(
        f"Evaluating: "
        f"{model_name}"
    )

    print("=" * 60)

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        print(
            f"Model not found: "
            f"{model_path}"
        )

        return None

    try:

        model = joblib.load(
            model_path
        )

    except Exception as error:

        print(
            f"Could not load "
            f"{model_name}:"
        )

        print(error)

        return None

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    start_time = (
        time.perf_counter()
    )

    try:

        predictions = model.predict(
            X_test
        )

    except Exception as error:

        print(
            f"Prediction failed "
            f"for {model_name}:"
        )

        print(error)

        return None

    prediction_time = (
        time.perf_counter()
        - start_time
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mape = calculate_mape(
        y_test,
        predictions
    )

    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    result = {

        "Model": model_name,

        "MAE": round(
            float(mae),
            4
        ),

        "RMSE": round(
            float(rmse),
            4
        ),

        "R2": round(
            float(r2),
            4
        ),

        "MAPE (%)": round(
            float(mape),
            4
        ),

        "Prediction Time (sec)": round(
            float(prediction_time),
            6
        ),

        "Model Path": str(
            model_path
        )
    }

    print(
        f"MAE       : "
        f"{result['MAE']}"
    )

    print(
        f"RMSE      : "
        f"{result['RMSE']}"
    )

    print(
        f"R2        : "
        f"{result['R2']}"
    )

    print(
        f"MAPE      : "
        f"{result['MAPE (%)']}%"
    )

    print(
        f"Prediction Time: "
        f"{result['Prediction Time (sec)']} sec"
    )

    return result


# =====================================================
# Compare Models
# =====================================================

def compare_models():

    print("\n" + "=" * 70)
    print("AQI MODEL COMPARISON")
    print("=" * 70)

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    df = load_dataset()

    print(
        f"Date Range: "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    if "city" in df.columns:

        print(
            f"Cities: "
            f"{df['city'].nunique()}"
        )

    # -------------------------------------------------
    # Features / Target
    # -------------------------------------------------

    X, y = split_features_target(
        df
    )

    print(
        f"\nFeatures Shape: "
        f"{X.shape}"
    )

    print(
        f"Target Shape: "
        f"{y.shape}"
    )

    # -------------------------------------------------
    # Chronological Split
    # -------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X,
        y
    )

    print(
        "\nTemporal split "
        "completed successfully."
    )

    print(
        "\nTest Dataset:"
    )

    print(
        f"Samples: "
        f"{len(X_test)}"
    )

    # -------------------------------------------------
    # Evaluate All Models
    # -------------------------------------------------

    results = []

    for model_name in MODEL_NAMES:

        model_path = (
            MODEL_DIR
            / f"{model_name}.pkl"
        )

        result = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            X_test=X_test,
            y_test=y_test,
        )

        if result is not None:

            results.append(
                result
            )

    if not results:

        raise RuntimeError(
            "No trained models "
            "were found."
        )

    # -------------------------------------------------
    # Comparison DataFrame
    # -------------------------------------------------

    comparison_df = pd.DataFrame(
        results
    )

    # Lower RMSE = better
    comparison_df = (
        comparison_df
        .sort_values(
            by="RMSE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    comparison_df.insert(
        0,
        "Rank",
        range(
            1,
            len(comparison_df) + 1
        )
    )

    # -------------------------------------------------
    # Save Comparison
    # -------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_path = (
        RESULTS_DIR
        / "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # -------------------------------------------------
    # Display Leaderboard
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL LEADERBOARD")
    print("=" * 70)

    print(
        comparison_df[
            [
                "Rank",
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "MAPE (%)",
                "Prediction Time (sec)",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Select Best Model
    # -------------------------------------------------

    best_model = (
        comparison_df.iloc[0]
    )

    best_model_name = (
        best_model["Model"]
    )

    best_model_path = (
        best_model["Model Path"]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model : "
        f"{best_model_name}"
    )

    print(
        f"RMSE  : "
        f"{best_model['RMSE']}"
    )

    print(
        f"MAE   : "
        f"{best_model['MAE']}"
    )

    print(
        f"R2    : "
        f"{best_model['R2']}"
    )

    print(
        f"MAPE  : "
        f"{best_model['MAPE (%)']}%"
    )

    print(
        "\nBest Model Path:"
    )

    print(
        best_model_path
    )

    # -------------------------------------------------
    # Dataset Version
    # -------------------------------------------------

    dataset_version = (
        get_dataset_version(
            df
        )
    )

    print(
        f"\nDataset Version: "
        f"{dataset_version}"
    )

    # -------------------------------------------------
    # Metrics For Registry
    # -------------------------------------------------

    metrics = {

        "MAE": float(
            best_model["MAE"]
        ),

        "RMSE": float(
            best_model["RMSE"]
        ),

        "R2": float(
            best_model["R2"]
        ),

        "MAPE": float(
            best_model["MAPE (%)"]
        )
    }

    # -------------------------------------------------
    # Register + Promotion
    # -------------------------------------------------

    registered_model = (
        register_and_promote_if_better(

            model_name=best_model_name,

            model_path=best_model_path,

            metrics=metrics,

            dataset_version=dataset_version,
        )
    )

    # -------------------------------------------------
    # Final Registry Result
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("REGISTRY RESULT")
    print("=" * 70)

    print(
        f"Model   : "
        f"{registered_model['model_name']}"
    )

    print(
        f"Version : "
        f"{registered_model['version']}"
    )

    print(
        f"Status  : "
        f"{registered_model['status']}"
    )

    print(
        f"RMSE    : "
        f"{registered_model['metrics']['RMSE']}"
    )

    print(
        f"Path    : "
        f"{registered_model['model_path']}"
    )

    print(
        "\nComparison saved to:"
    )

    print(
        comparison_path
    )

    print(
        "\nModel registry updated."
    )

    return comparison_df


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    try:

        compare_models()

    except Exception as error:

        print("\n" + "=" * 70)
        print(
            "ERROR: MODEL COMPARISON "
            "OR REGISTRATION FAILED"
        )
        print("=" * 70)

        raise