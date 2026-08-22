"""
evaluate_models.py

Evaluate all trained AQI prediction models.

The models are saved by train_models.py as complete
preprocessing + model pipelines in .joblib files.

This script:
    1. Loads the cached AQI dataset.
    2. Creates the same chronological train/test split.
    3. Loads each trained .joblib pipeline using joblib.
    4. Passes the original test DataFrame directly to the pipeline.
    5. Calculates MAE, RMSE and R².
    6. Selects the best model based on RMSE.
    7. Saves evaluation results and best model information.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.model_training.config import (
    MODEL_DIR,
    TARGET_COLUMN,
)

from src.model_training.utils import (
    load_dataset,
    split_features_target,
    split_dataset,
    load_model,
)


# ============================================================
# MODEL NAMES
# ============================================================

MODEL_NAMES = [
    "Ridge",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


# ============================================================
# MODEL FILE EXTENSION
# ============================================================

MODEL_EXTENSION = ".joblib"


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):
    """
    Evaluate one complete preprocessing + model pipeline.

    Parameters
    ----------
    model_name : str
        Name of the model.

    model :
        Loaded sklearn-compatible pipeline.

    X_test : pandas.DataFrame
        Original test features.

    y_test : pandas.Series
        Actual target values.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    predictions = np.asarray(predictions).reshape(-1)
    y_actual = np.asarray(y_test).reshape(-1)

    # --------------------------------------------------------
    # Validate prediction size
    # --------------------------------------------------------

    if len(predictions) != len(y_actual):
        raise ValueError(
            f"Prediction length mismatch.\n"
            f"Expected: {len(y_actual)}\n"
            f"Received: {len(predictions)}"
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_actual,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            predictions,
        )
    )

    r2 = r2_score(
        y_actual,
        predictions,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nEvaluation Metrics")
    print("-" * 40)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1: LOAD DATASET
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 1: LOADING DATASET")
    print("-" * 70)

    df = load_dataset()

    if df is None:
        raise RuntimeError(
            "Dataset loader returned None."
        )

    if df.empty:
        raise RuntimeError(
            "Dataset is empty."
        )

    print(
        f"\nDataset shape: {df.shape}"
    )

    # ========================================================
    # STEP 2: PREPARE FEATURES AND TARGET
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("-" * 70)

    X, y = split_features_target(df)

    print(
        f"\nFeatures Shape: {X.shape}"
    )

    print(
        f"Target Shape  : {y.shape}"
    )

    print(
        f"\nFeatures : {len(X.columns)}"
    )

    print(
        f"Target   : {TARGET_COLUMN}"
    )

    print(
        f"Rows     : {len(X):,}"
    )

    # ========================================================
    # STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 3: CREATING CHRONOLOGICAL TEST SET")
    print("-" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        X,
        y,
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

    # --------------------------------------------------------
    # Important
    # --------------------------------------------------------
    #
    # DO NOT manually encode X_test.
    #
    # train_models.py saved the preprocessing and model
    # together as one pipeline.
    #
    # Therefore:
    #
    #     X_test
    #        ↓
    #   saved pipeline
    #        ↓
    #   preprocessing
    #        ↓
    #      model
    #        ↓
    #   predictions
    #
    # --------------------------------------------------------

    print(
        "\nUsing original test DataFrame."
    )

    print(
        "Saved pipelines will perform preprocessing automatically."
    )

    # ========================================================
    # STEP 4: CHECK MODEL DIRECTORY
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 4: CHECKING TRAINED MODELS")
    print("-" * 70)

    model_directory = Path(MODEL_DIR)

    if not model_directory.exists():

        raise FileNotFoundError(
            f"Model directory does not exist:\n"
            f"{model_directory}"
        )

    print(
        f"\nModel directory:\n"
        f"{model_directory}"
    )

    # ========================================================
    # STEP 5: EVALUATE ALL MODELS
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("-" * 70)

    results = []

    for model_name in MODEL_NAMES:

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        # ----------------------------------------------------
        # Model path
        # ----------------------------------------------------

        model_path = (
            model_directory
            / f"{model_name}{MODEL_EXTENSION}"
        )

        print(
            f"\nLooking for:\n"
            f"{model_path}"
        )

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not model_path.exists():

            print(
                "\nWARNING:"
            )

            print(
                f"Model file not found:\n"
                f"{model_path}"
            )

            continue

        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        file_size_mb = (
            model_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"\nModel file found."
        )

        print(
            f"File size: "
            f"{file_size_mb:.2f} MB"
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        try:

            print(
                "\nLoading model using joblib..."
            )

            model = load_model(
                model_path
            )

            print(
                "Model loaded successfully."
            )

        except Exception as error:

            print(
                "\nERROR loading model:"
            )

            print(
                repr(error)
            )

            continue

        # ----------------------------------------------------
        # Evaluate model
        # ----------------------------------------------------

        try:

            result = evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            results.append(
                result
            )

            print(
                f"\n{model_name} evaluation completed."
            )

        except Exception as error:

            print(
                f"\nERROR evaluating {model_name}:"
            )

            print(
                repr(error)
            )

            continue

    # ========================================================
    # STEP 6: CHECK RESULTS
    # ========================================================

    if not results:

        raise RuntimeError(
            "\nNo models could be evaluated.\n\n"
            "Check that the .joblib files exist in:\n"
            f"{model_directory}\n\n"
            "Expected files:\n"
            + "\n".join(
                f"  - {name}.joblib"
                for name in MODEL_NAMES
            )
        )

    # ========================================================
    # STEP 7: CREATE RESULTS DATAFRAME
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 6: CREATING EVALUATION RESULTS")
    print("-" * 70)

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by RMSE
    # Lower RMSE = better
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            by="RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add ranking
    # --------------------------------------------------------

    results_df.insert(
        0,
        "Rank",
        range(
            1,
            len(results_df) + 1,
        ),
    )

    # ========================================================
    # STEP 8: DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # STEP 9: SELECT BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    best_model = results_df.iloc[0]

    best_model_name = (
        str(best_model["Model"])
    )

    best_mae = (
        float(best_model["MAE"])
    )

    best_rmse = (
        float(best_model["RMSE"])
    )

    best_r2 = (
        float(best_model["R2"])
    )

    print(
        f"\nModel : {best_model_name}"
    )

    print(
        f"MAE   : {best_mae:.4f}"
    )

    print(
        f"RMSE  : {best_rmse:.4f}"
    )

    print(
        f"R²    : {best_r2:.4f}"
    )

    # ========================================================
    # STEP 10: SAVE EVALUATION RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 7: SAVING EVALUATION RESULTS")
    print("-" * 70)

    evaluation_path = (
        model_directory
        / "evaluation_results.csv"
    )

    results_df.to_csv(
        evaluation_path,
        index=False,
    )

    print(
        f"\nEvaluation results saved to:"
    )

    print(
        evaluation_path
    )

    # ========================================================
    # STEP 11: SAVE BEST MODEL NAME
    # ========================================================

    best_model_path = (
        model_directory
        / "best_model.txt"
    )

    best_model_path.write_text(
        best_model_name,
        encoding="utf-8",
    )

    print(
        f"\nBest model name saved to:"
    )

    print(
        best_model_path
    )

    # ========================================================
    # STEP 12: SAVE EVALUATION METADATA
    # ========================================================

    evaluation_metadata = {
        "best_model": best_model_name,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "best_r2": best_r2,
        "models_evaluated": len(results_df),
        "test_samples": len(X_test),
        "training_samples": len(X_train),
        "target_column": TARGET_COLUMN,
        "model_extension": MODEL_EXTENSION,
    }

    metadata_path = (
        model_directory
        / "evaluation_metadata.csv"
    )

    metadata_df = pd.DataFrame(
        [
            evaluation_metadata
        ]
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    print(
        f"\nEvaluation metadata saved to:"
    )

    print(
        metadata_path
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nModels evaluated : "
        f"{len(results_df)}"
    )

    print(
        f"Best model      : "
        f"{best_model_name}"
    )

    print(
        f"Best RMSE       : "
        f"{best_rmse:.4f}"
    )

    print(
        f"Best MAE        : "
        f"{best_mae:.4f}"
    )

    print(
        f"Best R²         : "
        f"{best_r2:.4f}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  ✓ {evaluation_path.name}"
    )

    print(
        f"  ✓ {best_model_path.name}"
    )

    print(
        f"  ✓ {metadata_path.name}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()