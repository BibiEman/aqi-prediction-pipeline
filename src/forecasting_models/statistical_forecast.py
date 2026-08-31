"""
Statistical AQI Forecasting Baseline
====================================

Evaluates an ARIMA model as a statistical forecasting baseline for the
AQI prediction project.

The experiment:
1. Loads the processed historical dataset.
2. Sorts observations chronologically for each city.
3. Uses the final observations as a holdout test period.
4. Fits an ARIMA model using only historical AQI values.
5. Forecasts the holdout period.
6. Calculates MAE, RMSE, and R2.
7. Saves per-city and overall evaluation results.

This model is experimental and does not replace the production
LightGBM forecasting model.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_training_dataset.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_comparison"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "arima_evaluation_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "arima_predictions.csv"


# ARIMA configuration.
# (1, 1, 1) is intentionally used as a simple statistical baseline.
ARIMA_ORDER = (1, 1, 1)

# Approximately 20% of each city's observations are used for testing.
TEST_FRACTION = 0.20


def load_dataset() -> pd.DataFrame:
    """Load and validate the processed AQI dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset was not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "timestamp",
        "city",
        "target_aqi",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["target_aqi"] = pd.to_numeric(
        df["target_aqi"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "city", "target_aqi"]
    )

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    print("Dataset loaded successfully")
    print(f"Rows: {len(df):,}")
    print(f"Cities: {df['city'].nunique()}")
    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    return df


def chronological_split(
    city_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one city's data chronologically.

    Earlier observations are used for training and the final
    observations are reserved for testing.
    """

    city_df = city_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    split_index = int(
        len(city_df) * (1 - TEST_FRACTION)
    )

    train_df = city_df.iloc[:split_index].copy()
    test_df = city_df.iloc[split_index:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty "
            "training or testing dataset."
        )

    return train_df, test_df


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def evaluate_city(
    city: str,
    city_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Train and evaluate ARIMA for one city."""

    print()
    print("=" * 70)
    print(f"City: {city}")
    print("=" * 70)

    train_df, test_df = chronological_split(
        city_df
    )

    train_series = (
        train_df["target_aqi"]
        .astype(float)
        .to_numpy()
    )

    test_series = (
        test_df["target_aqi"]
        .astype(float)
        .to_numpy()
    )

    print(
        f"Training observations: "
        f"{len(train_series):,}"
    )
    print(
        f"Testing observations: "
        f"{len(test_series):,}"
    )
    print(
        f"ARIMA order: {ARIMA_ORDER}"
    )

    model = ARIMA(
        train_series,
        order=ARIMA_ORDER,
    )

    fitted_model = model.fit()

    predictions = fitted_model.forecast(
        steps=len(test_series)
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    metrics = calculate_metrics(
        test_series,
        predictions,
    )

    print(
        f"MAE:  {metrics['MAE']:.4f}"
    )
    print(
        f"RMSE: {metrics['RMSE']:.4f}"
    )
    print(
        f"R2:   {metrics['R2']:.4f}"
    )

    result = {
        "Model": "ARIMA",
        "City": city,
        "Order": str(ARIMA_ORDER),
        "Train_Rows": len(train_series),
        "Test_Rows": len(test_series),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }

    prediction_df = pd.DataFrame(
        {
            "timestamp": test_df[
                "timestamp"
            ].to_numpy(),
            "city": city,
            "actual_aqi": test_series,
            "predicted_aqi": predictions,
        }
    )

    return result, prediction_df


def calculate_overall_metrics(
    predictions_df: pd.DataFrame,
) -> dict:
    """Calculate metrics across all city predictions."""

    metrics = calculate_metrics(
        predictions_df[
            "actual_aqi"
        ].to_numpy(),
        predictions_df[
            "predicted_aqi"
        ].to_numpy(),
    )

    return {
        "Model": "ARIMA",
        "City": "Overall",
        "Order": str(ARIMA_ORDER),
        "Train_Rows": np.nan,
        "Test_Rows": len(predictions_df),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }


def main() -> None:
    """Run the complete ARIMA evaluation."""

    print("=" * 70)
    print("STATISTICAL AQI FORECASTING BASELINE")
    print("=" * 70)

    print(
        "\nThis experiment evaluates ARIMA as a "
        "statistical forecasting baseline."
    )
    print(
        "The production LightGBM model is not modified."
    )

    df = load_dataset()

    evaluation_results = []
    prediction_frames = []

    cities = sorted(
        df["city"].unique()
    )

    for city in cities:
        city_df = df[
            df["city"] == city
        ].copy()

        try:
            result, predictions = evaluate_city(
                city,
                city_df,
            )

            evaluation_results.append(
                result
            )

            prediction_frames.append(
                predictions
            )

        except Exception as exc:
            print()
            print(
                f"ARIMA evaluation failed for "
                f"{city}: {exc}"
            )

    if not prediction_frames:
        raise RuntimeError(
            "ARIMA evaluation failed for all cities."
        )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    overall_result = calculate_overall_metrics(
        predictions_df
    )

    evaluation_results.append(
        overall_result
    )

    results_df = pd.DataFrame(
        evaluation_results
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("OVERALL ARIMA RESULTS")
    print("=" * 70)

    print(
        f"MAE:  "
        f"{overall_result['MAE']:.4f}"
    )
    print(
        f"RMSE: "
        f"{overall_result['RMSE']:.4f}"
    )
    print(
        f"R2:   "
        f"{overall_result['R2']:.4f}"
    )

    print()
    print("Per-city results:")
    print(
        results_df[
            [
                "City",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Evaluation results saved to:"
    )
    print(RESULTS_PATH)

    print()
    print(
        "Predictions saved to:"
    )
    print(PREDICTIONS_PATH)

    print()
    print("=" * 70)
    print("ARIMA EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()