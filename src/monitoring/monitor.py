"""
monitor.py

Production monitoring for the AQI Prediction Pipeline.

Responsibilities
----------------
1. Load the production model from the model registry.
2. Load the model-training dataset.
3. Split the dataset chronologically into reference/current periods.
4. Detect numerical feature drift using PSI.
5. Detect categorical distribution drift.
6. Evaluate the production model on recent labelled data.
7. Determine overall model-health status.
8. Save monitoring reports.
9. Save monitoring summary.
10. Append monitoring history.

Outputs
-------
results/monitoring/
    data_drift/
        drift_report.csv
        categorical_drift_report.csv

    summary/
        monitoring_summary.json

    history/
        monitoring_history.csv
"""

import json
from datetime import datetime, timezone
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
    DATASET_PATH,
    REGISTRY_FILE,
    RESULTS_DIR,
    TARGET_COLUMN,
    DROP_COLUMNS,
    DRIFT_THRESHOLD,
    RMSE_ALERT_THRESHOLD,
)

from src.monitoring.drift_detection import (
    PSI_NO_DRIFT,
    PSI_MODERATE_DRIFT,
    detect_feature_drift,
    calculate_categorical_drift,
    summarize_numeric_drift,
    summarize_categorical_drift,
)


# ============================================================
# Configuration
# ============================================================

MONITORING_DIR = (
    Path(RESULTS_DIR)
    / "monitoring"
)

DRIFT_DIR = (
    MONITORING_DIR
    / "data_drift"
)

SUMMARY_DIR = (
    MONITORING_DIR
    / "summary"
)

HISTORY_DIR = (
    MONITORING_DIR
    / "history"
)


NUMERIC_DRIFT_REPORT_FILE = (
    DRIFT_DIR
    / "drift_report.csv"
)

CATEGORICAL_DRIFT_REPORT_FILE = (
    DRIFT_DIR
    / "categorical_drift_report.csv"
)

SUMMARY_FILE = (
    SUMMARY_DIR
    / "monitoring_summary.json"
)

HISTORY_FILE = (
    HISTORY_DIR
    / "monitoring_history.csv"
)


# ============================================================
# Monitoring Split
# ============================================================

# First 80% = reference
# Last 20% = current/recent data

REFERENCE_RATIO = 0.80


# ============================================================
# Overall Drift Rule
# ============================================================

# This value is used by the monitoring stage as an additional
# indicator based on the mean PSI of numerical features.

CONFIGURED_DRIFT_THRESHOLD = float(
    DRIFT_THRESHOLD
)


# ============================================================
# Directory Setup
# ============================================================

def create_directories():
    """
    Create monitoring output directories.
    """

    DRIFT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Registry Loading
# ============================================================

def load_registry():
    """
    Load registry.json.
    """

    registry_path = Path(
        REGISTRY_FILE
    )

    if not registry_path.exists():

        raise FileNotFoundError(
            "Model registry not found:\n"
            f"{registry_path}"
        )

    try:

        with open(
            registry_path,
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(
                file
            )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Invalid model registry JSON:\n"
            f"{registry_path}"
        ) from error

    return registry


# ============================================================
# Production Model Information
# ============================================================

def get_production_model_info(
    registry,
):
    """
    Find the model version currently marked as production.
    """

    models = registry.get(
        "models",
        {},
    )

    if not models:

        raise RuntimeError(
            "Model registry is empty."
        )

    for (
        model_name,
        model_info,
    ) in models.items():

        production_version = (
            model_info.get(
                "production_version"
            )
        )

        if not production_version:
            continue

        versions = model_info.get(
            "versions",
            [],
        )

        for version_info in versions:

            if (
                version_info.get(
                    "version"
                )
                == production_version
            ):

                model_path = (
                    version_info.get(
                        "model_path"
                    )
                )

                if not model_path:

                    raise RuntimeError(
                        "Production model path "
                        "is missing from registry."
                    )

                return {
                    "model_name":
                        model_name,

                    "version":
                        production_version,

                    "model_path":
                        model_path,

                    "metrics":
                        version_info.get(
                            "metrics",
                            {},
                        ),

                    "dataset_version":
                        version_info.get(
                            "dataset_version"
                        ),

                    "registered_at":
                        version_info.get(
                            "registered_at"
                        ),

                    "status":
                        version_info.get(
                            "status",
                            "production",
                        ),
                }

    raise RuntimeError(
        "No production model was found "
        "in the registry."
    )


# ============================================================
# Load Production Model
# ============================================================

def load_production_model(
    production_info,
):
    """
    Load the registered production model using joblib.
    """

    model_path = Path(
        production_info[
            "model_path"
        ]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Production model artifact "
            "does not exist:\n"
            f"{model_path}"
        )

    print(
        "\nLoading production model..."
    )

    print(
        f"Model   : "
        f"{production_info['model_name']}"
    )

    print(
        f"Version : "
        f"{production_info['version']}"
    )

    print(
        f"Path    : "
        f"{model_path}"
    )

    try:

        model = joblib.load(
            model_path
        )

    except Exception as error:

        raise RuntimeError(
            "Could not load production model.\n"
            f"Error: {error}"
        ) from error

    print(
        "\nProduction model loaded successfully."
    )

    return model


# ============================================================
# Load Monitoring Dataset
# ============================================================

def load_monitoring_dataset():
    """
    Load the model-training dataset used for monitoring.
    """

    dataset_path = Path(
        DATASET_PATH
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            "Monitoring dataset not found:\n"
            f"{dataset_path}"
        )

    print(
        "\nLoading monitoring dataset..."
    )

    df = pd.read_csv(
        dataset_path
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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
            "Monitoring dataset is missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        invalid_count = int(
            df["timestamp"]
            .isna()
            .sum()
        )

        raise ValueError(
            "Invalid timestamps found in "
            "monitoring dataset: "
            f"{invalid_count}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if df[TARGET_COLUMN].isna().any():

        print(
            "\nWarning:"
        )

        print(
            "Some target values are missing. "
            "Rows without targets will be excluded "
            "from performance evaluation."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nMonitoring dataset loaded."
    )

    print(
        f"Rows       : "
        f"{len(df):,}"
    )

    print(
        f"Columns    : "
        f"{len(df.columns)}"
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

    return df


# ============================================================
# Chronological Reference / Current Split
# ============================================================

def split_reference_current(
    df,
):
    """
    Split monitoring data chronologically.

    Reference
    ---------
    Older 80% of observations.

    Current
    -------
    Most recent 20% of observations.
    """

    if len(df) < 10:

        raise ValueError(
            "Monitoring dataset is too small."
        )

    split_index = int(
        len(df)
        * REFERENCE_RATIO
    )

    if split_index <= 0:

        raise RuntimeError(
            "Reference dataset would be empty."
        )

    if split_index >= len(df):

        raise RuntimeError(
            "Current monitoring dataset "
            "would be empty."
        )

    reference = (
        df.iloc[
            :split_index
        ]
        .copy()
    )

    current = (
        df.iloc[
            split_index:
        ]
        .copy()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REFERENCE / CURRENT MONITORING SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nReference Samples : "
        f"{len(reference):,}"
    )

    print(
        f"Reference Range   : "
        f"{reference['timestamp'].min()} "
        f"→ "
        f"{reference['timestamp'].max()}"
    )

    print(
        f"\nCurrent Samples   : "
        f"{len(current):,}"
    )

    print(
        f"Current Range     : "
        f"{current['timestamp'].min()} "
        f"→ "
        f"{current['timestamp'].max()}"
    )

    return (
        reference,
        current,
    )


# ============================================================
# Prepare Model Features
# ============================================================

def prepare_model_features(
    dataframe,
):
    """
    Build the same raw feature DataFrame supplied
    to the complete sklearn model pipeline.
    """

    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in dataframe.columns
    ]

    X = dataframe.drop(
        columns=columns_to_drop
    ).copy()

    return X


# ============================================================
# Evaluate Recent Model Performance
# ============================================================

def evaluate_current_performance(
    model,
    current,
    production_info,
):
    """
    Evaluate production model on the current/recent dataset.

    Because target_aqi is available in the historical monitoring
    dataset, this provides an actual recent RMSE rather than only
    reading the registry RMSE.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CURRENT MODEL PERFORMANCE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    evaluation_data = (
        current[
            current[
                TARGET_COLUMN
            ].notna()
        ]
        .copy()
    )

    if evaluation_data.empty:

        print(
            "\nCurrent labelled data is unavailable."
        )

        return {
            "available": False,

            "rmse": None,

            "mae": None,

            "r2": None,

            "registered_rmse":
                production_info.get(
                    "metrics",
                    {},
                ).get(
                    "RMSE"
                ),

            "threshold":
                float(
                    RMSE_ALERT_THRESHOLD
                ),

            "status":
                "UNAVAILABLE",
        }

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X_current = prepare_model_features(
        evaluation_data
    )

    y_current = (
        evaluation_data[
            TARGET_COLUMN
        ]
        .astype(float)
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            X_current
        )

    except Exception as error:

        print(
            "\nPerformance evaluation failed:"
        )

        print(
            error
        )

        return {
            "available": False,

            "rmse": None,

            "mae": None,

            "r2": None,

            "registered_rmse":
                production_info.get(
                    "metrics",
                    {},
                ).get(
                    "RMSE"
                ),

            "threshold":
                float(
                    RMSE_ALERT_THRESHOLD
                ),

            "status":
                "UNAVAILABLE",
        }

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = float(
        mean_absolute_error(
            y_current,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_current,
                predictions,
            )
        )
    )

    r2 = float(
        r2_score(
            y_current,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Registered baseline RMSE
    # --------------------------------------------------------

    registered_rmse = (
        production_info.get(
            "metrics",
            {},
        ).get(
            "RMSE"
        )
    )

    try:

        registered_rmse = (
            float(
                registered_rmse
            )
            if registered_rmse is not None
            else None
        )

    except (
        TypeError,
        ValueError,
    ):

        registered_rmse = None

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    threshold = float(
        RMSE_ALERT_THRESHOLD
    )

    if rmse >= threshold:

        status = "ALERT"

    else:

        status = "NORMAL"

    # --------------------------------------------------------
    # Performance change
    # --------------------------------------------------------

    rmse_change = None

    rmse_change_percentage = None

    if (
        registered_rmse is not None
        and registered_rmse > 0
    ):

        rmse_change = (
            rmse
            - registered_rmse
        )

        rmse_change_percentage = (
            rmse_change
            / registered_rmse
            * 100
        )

    print(
        f"\nCurrent Samples      : "
        f"{len(evaluation_data):,}"
    )

    print(
        f"Current MAE          : "
        f"{mae:.4f}"
    )

    print(
        f"Current RMSE         : "
        f"{rmse:.4f}"
    )

    print(
        f"Current R²           : "
        f"{r2:.4f}"
    )

    print(
        f"Registered RMSE      : "
        f"{registered_rmse}"
    )

    print(
        f"RMSE Alert Threshold : "
        f"{threshold}"
    )

    print(
        f"Performance Status   : "
        f"{status}"
    )

    if rmse_change_percentage is not None:

        print(
            f"RMSE Change          : "
            f"{rmse_change_percentage:.2f}%"
        )

    return {
        "available":
            True,

        "samples":
            int(
                len(
                    evaluation_data
                )
            ),

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "registered_rmse":
            registered_rmse,

        "rmse_change":
            rmse_change,

        "rmse_change_percentage":
            rmse_change_percentage,

        "threshold":
            threshold,

        "status":
            status,
    }


# ============================================================
# Mean PSI
# ============================================================

def calculate_mean_psi(
    drift_report,
):
    """
    Calculate mean finite PSI across numerical features.
    """

    if drift_report.empty:

        return None

    psi_values = pd.to_numeric(
        drift_report[
            "PSI"
        ],
        errors="coerce",
    )

    psi_values = psi_values[
        np.isfinite(
            psi_values
        )
    ]

    if psi_values.empty:

        return None

    return float(
        psi_values.mean()
    )


# ============================================================
# Determine Overall Status
# ============================================================

def determine_overall_status(
    numeric_summary,
    categorical_summary,
    performance,
    mean_psi,
):
    """
    Determine overall health status.

    Priority
    --------
    ALERT
        Performance threshold exceeded.

    DRIFT_DETECTED
        At least one severe numerical or categorical drift.

    MODERATE_DRIFT
        Moderate drift exists or mean PSI exceeds the
        configured general drift threshold.

    HEALTHY
        No actionable problems.
    """

    if (
        performance.get(
            "status"
        )
        == "ALERT"
    ):

        return "ALERT"

    if (
        numeric_summary.get(
            "drift_features",
            0,
        ) > 0
        or
        categorical_summary.get(
            "drift_features",
            0,
        ) > 0
    ):

        return "DRIFT_DETECTED"

    if (
        numeric_summary.get(
            "moderate_drift_features",
            0,
        ) > 0
        or
        categorical_summary.get(
            "moderate_drift_features",
            0,
        ) > 0
    ):

        return "MODERATE_DRIFT"

    if (
        mean_psi is not None
        and
        mean_psi
        >= CONFIGURED_DRIFT_THRESHOLD
    ):

        return "MODERATE_DRIFT"

    return "HEALTHY"


# ============================================================
# Save Numeric Drift Report
# ============================================================

def save_numeric_drift_report(
    report,
):
    """
    Save numerical feature PSI report.
    """

    report.to_csv(
        NUMERIC_DRIFT_REPORT_FILE,
        index=False,
    )

    print(
        "\nNumerical drift report saved:"
    )

    print(
        NUMERIC_DRIFT_REPORT_FILE
    )


# ============================================================
# Save Categorical Drift Report
# ============================================================

def save_categorical_drift_report(
    report,
):
    """
    Save categorical drift report.
    """

    report.to_csv(
        CATEGORICAL_DRIFT_REPORT_FILE,
        index=False,
    )

    print(
        "\nCategorical drift report saved:"
    )

    print(
        CATEGORICAL_DRIFT_REPORT_FILE
    )


# ============================================================
# Save Monitoring Summary
# ============================================================

def save_summary(
    summary,
):
    """
    Save monitoring summary JSON.
    """

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )

    print(
        "\nMonitoring summary saved:"
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# Save Monitoring History
# ============================================================

def save_monitoring_history(
    summary,
):
    """
    Append current monitoring run to history CSV.
    """

    model_performance = (
        summary.get(
            "model_performance",
            {},
        )
    )

    feature_drift = (
        summary.get(
            "feature_drift",
            {},
        )
    )

    categorical_drift = (
        summary.get(
            "categorical_drift",
            {},
        )
    )

    history_row = {
        "timestamp":
            summary.get(
                "monitoring_timestamp"
            ),

        "model":
            summary[
                "production_model"
            ].get(
                "model_name"
            ),

        "version":
            summary[
                "production_model"
            ].get(
                "version"
            ),

        "overall_status":
            summary.get(
                "overall_status"
            ),

        "numeric_features":
            feature_drift.get(
                "numeric_features"
            ),

        "drift_features":
            feature_drift.get(
                "drift_features"
            ),

        "moderate_drift_features":
            feature_drift.get(
                "moderate_drift_features"
            ),

        "no_drift_features":
            feature_drift.get(
                "no_drift_features"
            ),

        "drift_percentage":
            feature_drift.get(
                "drift_percentage"
            ),

        "mean_psi":
            feature_drift.get(
                "mean_psi"
            ),

        "categorical_drift_features":
            categorical_drift.get(
                "drift_features"
            ),

        "current_mae":
            model_performance.get(
                "mae"
            ),

        "current_rmse":
            model_performance.get(
                "rmse"
            ),

        "current_r2":
            model_performance.get(
                "r2"
            ),

        "registered_rmse":
            model_performance.get(
                "registered_rmse"
            ),

        "rmse_alert_threshold":
            model_performance.get(
                "threshold"
            ),

        "performance_status":
            model_performance.get(
                "status"
            ),
    }

    row_df = pd.DataFrame(
        [
            history_row
        ]
    )

    if HISTORY_FILE.exists():

        row_df.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        row_df.to_csv(
            HISTORY_FILE,
            index=False,
        )

    print(
        "\nMonitoring history updated:"
    )

    print(
        HISTORY_FILE
    )


# ============================================================
# Display Drift Report
# ============================================================

def display_drift_results(
    numeric_report,
    categorical_report,
):
    """
    Print feature drift results.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "NUMERICAL FEATURE DRIFT"
    )

    print(
        "=" * 70
    )

    if numeric_report.empty:

        print(
            "\nNo numerical features "
            "were available for monitoring."
        )

    else:

        display_columns = [
            column
            for column in [
                "Feature",
                "PSI",
                "Status",
                "Reference Mean",
                "Current Mean",
            ]
            if column
            in numeric_report.columns
        ]

        print(
            "\nTop numerical drift results:"
        )

        print(
            numeric_report[
                display_columns
            ]
            .head(
                15
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "CATEGORICAL FEATURE DRIFT"
    )

    print(
        "=" * 70
    )

    if categorical_report.empty:

        print(
            "\nNo categorical features "
            "were available for monitoring."
        )

    else:

        print(
            "\n"
            + categorical_report
            .to_string(
                index=False
            )
        )


# ============================================================
# Main Monitoring Process
# ============================================================

def monitor():
    """
    Execute complete production monitoring process.
    """

    create_directories()

    print(
        "\n" + "=" * 70
    )

    print(
        "AQI PRODUCTION MODEL MONITORING"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # STEP 1: Registry
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 1: LOADING PRODUCTION MODEL"
    )

    print(
        "-" * 70
    )

    registry = load_registry()

    production_info = (
        get_production_model_info(
            registry
        )
    )

    model = load_production_model(
        production_info
    )


    # ========================================================
    # STEP 2: Dataset
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 2: LOADING MONITORING DATA"
    )

    print(
        "-" * 70
    )

    df = load_monitoring_dataset()


    # ========================================================
    # STEP 3: Reference / Current
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 3: CREATING MONITORING WINDOWS"
    )

    print(
        "-" * 70
    )

    (
        reference,
        current,
    ) = split_reference_current(
        df
    )


    # ========================================================
    # STEP 4: Numeric Drift
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 4: NUMERICAL FEATURE DRIFT"
    )

    print(
        "-" * 70
    )

    print(
        "\nCalculating PSI..."
    )

    numeric_report = (
        detect_feature_drift(
            reference=reference,
            current=current,
        )
    )


    # ========================================================
    # STEP 5: Categorical Drift
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 5: CATEGORICAL FEATURE DRIFT"
    )

    print(
        "-" * 70
    )

    categorical_report = (
        calculate_categorical_drift(
            reference=reference,
            current=current,
        )
    )


    # ========================================================
    # STEP 6: Summaries
    # ========================================================

    numeric_summary = (
        summarize_numeric_drift(
            numeric_report
        )
    )

    categorical_summary = (
        summarize_categorical_drift(
            categorical_report
        )
    )

    mean_psi = calculate_mean_psi(
        numeric_report
    )

    numeric_summary[
        "mean_psi"
    ] = (
        round(
            mean_psi,
            6,
        )
        if mean_psi is not None
        else None
    )

    numeric_summary[
        "thresholds"
    ] = {
        "no_drift":
            PSI_NO_DRIFT,

        "moderate_drift":
            PSI_MODERATE_DRIFT,

        "configured_drift_threshold":
            CONFIGURED_DRIFT_THRESHOLD,
    }


    # ========================================================
    # STEP 7: Model Performance
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 6: MODEL PERFORMANCE MONITORING"
    )

    print(
        "-" * 70
    )

    performance = (
        evaluate_current_performance(
            model=model,
            current=current,
            production_info=production_info,
        )
    )


    # ========================================================
    # STEP 8: Overall Status
    # ========================================================

    overall_status = (
        determine_overall_status(
            numeric_summary=(
                numeric_summary
            ),
            categorical_summary=(
                categorical_summary
            ),
            performance=(
                performance
            ),
            mean_psi=(
                mean_psi
            ),
        )
    )


    # ========================================================
    # STEP 9: Display Drift
    # ========================================================

    display_drift_results(
        numeric_report=(
            numeric_report
        ),
        categorical_report=(
            categorical_report
        ),
    )


    # ========================================================
    # STEP 10: Build Summary
    # ========================================================

    monitoring_timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    summary = {
        "monitoring_timestamp":
            monitoring_timestamp,

        "overall_status":
            overall_status,

        # ----------------------------------------------------
        # Production model
        # ----------------------------------------------------

        "production_model": {
            "model_name":
                production_info[
                    "model_name"
                ],

            "version":
                production_info[
                    "version"
                ],

            "model_path":
                production_info[
                    "model_path"
                ],

            "dataset_version":
                production_info.get(
                    "dataset_version"
                ),

            "registered_at":
                production_info.get(
                    "registered_at"
                ),

            "status":
                production_info.get(
                    "status"
                ),

            "metrics":
                production_info.get(
                    "metrics",
                    {},
                ),
        },

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "dataset": {
            "total_samples":
                int(
                    len(df)
                ),

            "reference_samples":
                int(
                    len(
                        reference
                    )
                ),

            "current_samples":
                int(
                    len(
                        current
                    )
                ),

            "reference_start":
                str(
                    reference[
                        "timestamp"
                    ].min()
                ),

            "reference_end":
                str(
                    reference[
                        "timestamp"
                    ].max()
                ),

            "current_start":
                str(
                    current[
                        "timestamp"
                    ].min()
                ),

            "current_end":
                str(
                    current[
                        "timestamp"
                    ].max()
                ),

            "cities":
                int(
                    df[
                        "city"
                    ].nunique()
                ),
        },

        # ----------------------------------------------------
        # Numeric feature drift
        # ----------------------------------------------------

        "feature_drift":
            numeric_summary,

        # ----------------------------------------------------
        # Categorical drift
        # ----------------------------------------------------

        "categorical_drift":
            categorical_summary,

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        "model_performance":
            performance,
    }


    # ========================================================
    # STEP 11: Save Outputs
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "STEP 7: SAVING MONITORING REPORTS"
    )

    print(
        "-" * 70
    )

    save_numeric_drift_report(
        numeric_report
    )

    save_categorical_drift_report(
        categorical_report
    )

    save_summary(
        summary
    )

    save_monitoring_history(
        summary
    )


    # ========================================================
    # Final Summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MONITORING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOverall Status       : "
        f"{overall_status}"
    )

    print(
        f"Production Model     : "
        f"{production_info['model_name']}"
    )

    print(
        f"Production Version   : "
        f"{production_info['version']}"
    )

    print(
        "\nFeature Drift"
    )

    print(
        "-" * 40
    )

    print(
        f"Numeric Features     : "
        f"{numeric_summary['numeric_features']}"
    )

    print(
        f"Drift Features       : "
        f"{numeric_summary['drift_features']}"
    )

    print(
        f"Moderate Drift       : "
        f"{numeric_summary['moderate_drift_features']}"
    )

    print(
        f"No Drift Features    : "
        f"{numeric_summary['no_drift_features']}"
    )

    print(
        f"Drift Percentage     : "
        f"{numeric_summary['drift_percentage']:.2f}%"
    )

    if mean_psi is not None:

        print(
            f"Mean PSI             : "
            f"{mean_psi:.6f}"
        )

    print(
        "\nCategorical Drift"
    )

    print(
        "-" * 40
    )

    print(
        f"Features Checked     : "
        f"{categorical_summary['features_checked']}"
    )

    print(
        f"Drift Features       : "
        f"{categorical_summary['drift_features']}"
    )

    print(
        f"Moderate Drift       : "
        f"{categorical_summary['moderate_drift_features']}"
    )

    print(
        "\nModel Performance"
    )

    print(
        "-" * 40
    )

    print(
        f"Current MAE          : "
        f"{performance.get('mae')}"
    )

    print(
        f"Current RMSE         : "
        f"{performance.get('rmse')}"
    )

    print(
        f"Current R²           : "
        f"{performance.get('r2')}"
    )

    print(
        f"Registered RMSE      : "
        f"{performance.get('registered_rmse')}"
    )

    print(
        f"RMSE Alert Threshold : "
        f"{performance.get('threshold')}"
    )

    print(
        f"Performance Status   : "
        f"{performance.get('status')}"
    )


    # ========================================================
    # Health Message
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    if overall_status == "HEALTHY":

        print(
            "✓ Production monitoring status: HEALTHY"
        )

        print(
            "✓ Production model remains acceptable."
        )

    elif overall_status == "MODERATE_DRIFT":

        print(
            "NOTICE:"
        )

        print(
            "Moderate feature drift was detected."
        )

        print(
            "Continue monitoring future observations."
        )

    elif overall_status == "DRIFT_DETECTED":

        print(
            "WARNING:"
        )

        print(
            "Significant feature drift was detected."
        )

        print(
            "Retraining assessment should be performed."
        )

    elif overall_status == "ALERT":

        print(
            "ALERT:"
        )

        print(
            "Production model performance exceeded "
            "the configured RMSE threshold."
        )

        print(
            "Retraining assessment should be performed."
        )


    # ========================================================
    # Files
    # ========================================================

    print(
        "\nGenerated Monitoring Files:"
    )

    print(
        f"  ✓ {NUMERIC_DRIFT_REPORT_FILE}"
    )

    print(
        f"  ✓ {CATEGORICAL_DRIFT_REPORT_FILE}"
    )

    print(
        f"  ✓ {SUMMARY_FILE}"
    )

    print(
        f"  ✓ {HISTORY_FILE}"
    )

    print(
        "\nMonitoring completed successfully."
    )

    print(
        "=" * 70
    )

    return summary


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    monitor()