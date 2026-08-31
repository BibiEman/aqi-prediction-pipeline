"""
Deep-Learning AQI Forecasting Baseline
======================================

Evaluates a Long Short-Term Memory (LSTM) neural network as a
deep-learning forecasting baseline for the AQI prediction project.

The experiment:
1. Loads the processed historical AQI dataset.
2. Processes each city independently.
3. Uses a chronological train/test split.
4. Fits the scaler only on training data to prevent leakage.
5. Converts historical AQI observations into sequences.
6. Trains an LSTM neural network.
7. Evaluates predictions using MAE, RMSE, and R2.
8. Saves per-city and overall results.

This is an experimental benchmark. It does not replace the production
LightGBM model.
"""

from pathlib import Path
import os
import random
import warnings

# Reduce TensorFlow console noise.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# Paths
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

RESULTS_PATH = RESULTS_DIR / "lstm_evaluation_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "lstm_predictions.csv"


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

# Number of historical observations supplied to the LSTM.
# With hourly historical data, 24 observations represent approximately
# one day of recent AQI history.
SEQUENCE_LENGTH = 24

TEST_FRACTION = 0.20

EPOCHS = 30
BATCH_SIZE = 32

LSTM_UNITS = 32

EARLY_STOPPING_PATIENCE = 5


def load_dataset() -> pd.DataFrame:
    """Load and validate the processed AQI dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at:\n{DATA_PATH}"
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
        subset=[
            "timestamp",
            "city",
            "target_aqi",
        ]
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


def create_sequences(
    values: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a time series into supervised learning sequences.

    X contains the previous sequence_length AQI observations.
    y contains the next AQI observation.
    """

    X = []
    y = []

    for index in range(
        sequence_length,
        len(values),
    ):
        X.append(
            values[
                index - sequence_length:index
            ]
        )

        y.append(
            values[index]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def build_lstm_model(
    sequence_length: int,
) -> Sequential:
    """Construct the LSTM neural network."""

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    1,
                )
            ),
            LSTM(
                LSTM_UNITS,
            ),
            Dense(
                16,
                activation="relu",
            ),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


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
    """Train and evaluate an LSTM for one city."""

    print()
    print("=" * 70)
    print(f"City: {city}")
    print("=" * 70)

    city_df = city_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    values = (
        city_df["target_aqi"]
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    split_index = int(
        len(values)
        * (1 - TEST_FRACTION)
    )

    if split_index <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough training observations "
            f"for {city}."
        )

    train_values = values[:split_index]

    # Include the final training sequence before the test boundary.
    # This allows the first test prediction to use only historical
    # observations that existed before that target observation.
    test_with_context = values[
        split_index - SEQUENCE_LENGTH:
    ]

    # -------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------
    # Fit ONLY on training data. This prevents test information from
    # influencing preprocessing.
    # -------------------------------------------------------------

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_values
    )

    test_scaled = scaler.transform(
        test_with_context
    )

    X_train, y_train = create_sequences(
        train_scaled,
        SEQUENCE_LENGTH,
    )

    X_test, y_test_scaled = create_sequences(
        test_scaled,
        SEQUENCE_LENGTH,
    )

    X_train = X_train.reshape(
        (
            X_train.shape[0],
            SEQUENCE_LENGTH,
            1,
        )
    )

    X_test = X_test.reshape(
        (
            X_test.shape[0],
            SEQUENCE_LENGTH,
            1,
        )
    )

    print(
        f"Training observations: "
        f"{len(train_values):,}"
    )
    print(
        f"Testing observations: "
        f"{len(values) - split_index:,}"
    )
    print(
        f"Training sequences: "
        f"{len(X_train):,}"
    )
    print(
        f"Testing sequences: "
        f"{len(X_test):,}"
    )
    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    # -------------------------------------------------------------
    # Model training
    # -------------------------------------------------------------

    model = build_lstm_model(
        SEQUENCE_LENGTH
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
        ],
        shuffle=False,
        verbose=0,
    )

    epochs_trained = len(
        history.history["loss"]
    )

    print(
        f"Epochs trained: "
        f"{epochs_trained}"
    )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    predicted_scaled = model.predict(
        X_test,
        verbose=0,
    )

    predictions = scaler.inverse_transform(
        predicted_scaled
    ).reshape(-1)

    actual_values = scaler.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).reshape(-1)

    metrics = calculate_metrics(
        actual_values,
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
        "Model": "LSTM",
        "City": city,
        "Sequence_Length": SEQUENCE_LENGTH,
        "Train_Rows": len(train_values),
        "Test_Rows": len(actual_values),
        "Epochs_Trained": epochs_trained,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }

    test_timestamps = city_df.iloc[
        split_index:
    ]["timestamp"].to_numpy()

    prediction_df = pd.DataFrame(
        {
            "timestamp": test_timestamps,
            "city": city,
            "actual_aqi": actual_values,
            "predicted_aqi": predictions,
        }
    )

    # Clear TensorFlow state between cities.
    tf.keras.backend.clear_session()

    return result, prediction_df


def calculate_overall_metrics(
    predictions_df: pd.DataFrame,
) -> dict:
    """Calculate overall LSTM evaluation metrics."""

    metrics = calculate_metrics(
        predictions_df[
            "actual_aqi"
        ].to_numpy(),
        predictions_df[
            "predicted_aqi"
        ].to_numpy(),
    )

    return {
        "Model": "LSTM",
        "City": "Overall",
        "Sequence_Length": SEQUENCE_LENGTH,
        "Train_Rows": np.nan,
        "Test_Rows": len(predictions_df),
        "Epochs_Trained": np.nan,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }


def main() -> None:
    """Run the complete LSTM experiment."""

    print("=" * 70)
    print("DEEP-LEARNING AQI FORECASTING BASELINE")
    print("=" * 70)

    print()
    print(
        "This experiment evaluates an LSTM "
        "as a deep-learning forecasting baseline."
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
                f"LSTM evaluation failed for "
                f"{city}: {exc}"
            )

            # Ensure TensorFlow state is cleared
            # even if one city's training fails.
            tf.keras.backend.clear_session()

    if not prediction_frames:
        raise RuntimeError(
            "LSTM evaluation failed for all cities."
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
    print("OVERALL LSTM RESULTS")
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
    print("LSTM EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()