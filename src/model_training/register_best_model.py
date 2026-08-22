"""
register_best_model.py

Register the best evaluated AQI model in the local model registry.

The best model is selected using the lowest RMSE from:
    models/evaluation_results.csv

The trained model is expected as:
    models/<ModelName>.joblib

Current evaluation metrics:
    - MAE
    - RMSE
    - R2
"""

from pathlib import Path

import pandas as pd

from src.model_training.config import (
    MODEL_DIR,
)

from src.model_training.model_registry import (
    register_and_promote_if_better,
)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(MODEL_DIR)

EVALUATION_FILE = (
    MODEL_DIR
    / "evaluation_results.csv"
)

BEST_MODEL_FILE = (
    MODEL_DIR
    / "best_model.txt"
)

DATASET_VERSION = "2025_Jan_Jun"


# ============================================================
# Load Evaluation Results
# ============================================================

def load_evaluation_results():
    """
    Load model evaluation results.
    """

    print("\n" + "=" * 70)
    print("LOADING MODEL EVALUATION RESULTS")
    print("=" * 70)

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            "\nEvaluation results were not found.\n"
            f"Expected:\n{EVALUATION_FILE}\n\n"
            "Run evaluate_models.py first."
        )

    metrics_df = pd.read_csv(
        EVALUATION_FILE
    )

    if metrics_df.empty:
        raise ValueError(
            "Evaluation results file is empty."
        )

    required_columns = [
        "Model",
        "MAE",
        "RMSE",
        "R2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Evaluation results are missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    print(
        f"\nModels found: "
        f"{len(metrics_df)}"
    )

    return metrics_df


# ============================================================
# Select Best Model
# ============================================================

def select_best_model(metrics_df):
    """
    Select the model with the lowest RMSE.
    """

    print("\n" + "=" * 70)
    print("SELECTING BEST MODEL")
    print("=" * 70)

    best_index = (
        metrics_df["RMSE"]
        .astype(float)
        .idxmin()
    )

    best = metrics_df.loc[
        best_index
    ]

    model_name = str(
        best["Model"]
    )

    print(
        f"\nBest Model : {model_name}"
    )

    print(
        f"MAE        : "
        f"{float(best['MAE']):.4f}"
    )

    print(
        f"RMSE       : "
        f"{float(best['RMSE']):.4f}"
    )

    print(
        f"R²         : "
        f"{float(best['R2']):.4f}"
    )

    return best


# ============================================================
# Validate Model Artifact
# ============================================================

def get_model_path(model_name):
    """
    Return and validate the trained model artifact.
    """

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "\nBest model artifact was not found.\n"
            f"Expected:\n{model_path}\n\n"
            "Run train_models.py first."
        )

    file_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"\nModel artifact:\n"
        f"{model_path}"
    )

    print(
        f"Model size: "
        f"{file_size_mb:.2f} MB"
    )

    return model_path


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI BEST MODEL REGISTRATION")
    print("=" * 70)

    # ========================================================
    # Step 1: Evaluation Results
    # ========================================================

    metrics_df = (
        load_evaluation_results()
    )

    # ========================================================
    # Step 2: Best Model
    # ========================================================

    best = select_best_model(
        metrics_df
    )

    model_name = str(
        best["Model"]
    )

    # ========================================================
    # Step 3: Model Artifact
    # ========================================================

    model_path = get_model_path(
        model_name
    )

    # ========================================================
    # Step 4: Metrics
    # ========================================================

    metrics = {
        "MAE": float(
            best["MAE"]
        ),
        "RMSE": float(
            best["RMSE"]
        ),
        "R2": float(
            best["R2"]
        ),

        # MAPE is not part of the current evaluator.
        # Kept only for compatibility with model_registry.py.
        "MAPE": 0.0,
    }

    # ========================================================
    # Step 5: Register
    # ========================================================

    print("\n" + "=" * 70)
    print("REGISTERING BEST MODEL")
    print("=" * 70)

    registered_model = (
        register_and_promote_if_better(
            model_name=model_name,
            model_path=model_path,
            metrics=metrics,
            dataset_version=DATASET_VERSION,
        )
    )

    # ========================================================
    # Step 6: Save Selected Model Name
    # ========================================================

    BEST_MODEL_FILE.write_text(
        model_name,
        encoding="utf-8",
    )

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL REGISTRATION COMPLETED")
    print("=" * 70)

    print(
        f"\nModel           : "
        f"{model_name}"
    )

    print(
        f"Source artifact : "
        f"{model_path}"
    )

    print(
        f"Dataset version : "
        f"{DATASET_VERSION}"
    )

    print(
        f"RMSE            : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"MAE             : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"R²              : "
        f"{metrics['R2']:.4f}"
    )

    if registered_model:

        print(
            f"Registry version: "
            f"{registered_model.get('version')}"
        )

        print(
            f"Status          : "
            f"{registered_model.get('status')}"
        )

    print(
        "\nBest-model registration "
        "finished successfully."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()