"""
retraining_trigger.py

Automated retraining decision for the AQI Prediction Pipeline.

Decision logic
--------------
1. Load the latest monitoring summary.
2. Inspect feature drift.
3. Inspect current model performance.
4. Compare current RMSE with registered baseline RMSE.
5. Retrain only when the monitoring evidence is actionable.
6. Do not retrain for seasonal drift alone when model
   performance remains stable.
"""

import json
import subprocess
import sys
from pathlib import Path

from src.model_training.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    RMSE_ALERT_THRESHOLD,
)


# ============================================================
# Configuration
# ============================================================

MONITORING_SUMMARY = (
    RESULTS_DIR
    / "monitoring"
    / "summary"
    / "monitoring_summary.json"
)


# Percentage of numeric features with significant drift
# before the drift condition is considered substantial.

DRIFT_PERCENTAGE_THRESHOLD = 30.0


# RMSE increase relative to registered baseline required
# before drift is considered performance-relevant.

RMSE_DEGRADATION_PERCENT_THRESHOLD = 10.0


# Existing training / model comparison module.

RETRAIN_MODULE = (
    "src.model_training.compare_models"
)


# ============================================================
# Load Monitoring Summary
# ============================================================

def load_monitoring_summary():
    """
    Load the latest monitoring summary.
    """

    if not MONITORING_SUMMARY.exists():

        raise FileNotFoundError(
            "Monitoring summary not found:\n"
            f"{MONITORING_SUMMARY}\n\n"
            "Run monitoring first:\n"
            "python -m src.monitoring.monitor"
        )

    try:

        with open(
            MONITORING_SUMMARY,
            "r",
            encoding="utf-8",
        ) as file:

            summary = json.load(
                file
            )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Monitoring summary contains "
            "invalid JSON."
        ) from error

    return summary


# ============================================================
# Safe Float
# ============================================================

def safe_float(
    value,
    default=None,
):
    """
    Convert a value to float safely.
    """

    if value is None:
        return default

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# Extract Monitoring Values
# ============================================================

def extract_monitoring_values(
    summary,
):
    """
    Extract values required for retraining decision.
    """

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_status = summary.get(
        "overall_status",
        "UNKNOWN",
    )


    # --------------------------------------------------------
    # Production model
    # --------------------------------------------------------

    production_model = summary.get(
        "production_model",
        {},
    )

    production_model_name = (
        production_model.get(
            "model_name",
            "UNKNOWN",
        )
    )

    production_version = (
        production_model.get(
            "version",
            "UNKNOWN",
        )
    )

    production_metrics = (
        production_model.get(
            "metrics",
            {},
        )
    )


    # --------------------------------------------------------
    # Feature drift
    # --------------------------------------------------------

    feature_drift = summary.get(
        "feature_drift",
        {},
    )

    numeric_features = int(
        feature_drift.get(
            "numeric_features",
            0,
        )
        or 0
    )

    drift_features = int(
        feature_drift.get(
            "drift_features",
            0,
        )
        or 0
    )

    moderate_drift_features = int(
        feature_drift.get(
            "moderate_drift_features",
            0,
        )
        or 0
    )

    no_drift_features = int(
        feature_drift.get(
            "no_drift_features",
            0,
        )
        or 0
    )

    drift_percentage = safe_float(
        feature_drift.get(
            "drift_percentage",
            0.0,
        ),
        0.0,
    )

    mean_psi = safe_float(
        feature_drift.get(
            "mean_psi"
        )
    )


    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    model_performance = summary.get(
        "model_performance",
        {},
    )

    performance_available = bool(
        model_performance.get(
            "available",
            False,
        )
    )

    current_rmse = safe_float(
        model_performance.get(
            "rmse"
        )
    )

    current_mae = safe_float(
        model_performance.get(
            "mae"
        )
    )

    current_r2 = safe_float(
        model_performance.get(
            "r2"
        )
    )

    registered_rmse = safe_float(
        model_performance.get(
            "registered_rmse",
            production_metrics.get(
                "RMSE"
            ),
        )
    )

    rmse_threshold = safe_float(
        model_performance.get(
            "threshold",
            RMSE_ALERT_THRESHOLD,
        ),
        float(
            RMSE_ALERT_THRESHOLD
        ),
    )

    performance_status = (
        model_performance.get(
            "status",
            "UNKNOWN",
        )
    )


    # --------------------------------------------------------
    # RMSE degradation
    # --------------------------------------------------------

    rmse_change = None

    rmse_change_percentage = None

    if (
        current_rmse is not None
        and registered_rmse is not None
        and registered_rmse > 0
    ):

        rmse_change = (
            current_rmse
            - registered_rmse
        )

        rmse_change_percentage = (
            rmse_change
            / registered_rmse
            * 100
        )


    return {
        "overall_status":
            overall_status,

        "production_model":
            production_model_name,

        "production_version":
            production_version,

        "numeric_features":
            numeric_features,

        "drift_features":
            drift_features,

        "moderate_drift_features":
            moderate_drift_features,

        "no_drift_features":
            no_drift_features,

        "drift_percentage":
            drift_percentage,

        "mean_psi":
            mean_psi,

        "performance_available":
            performance_available,

        "current_mae":
            current_mae,

        "current_rmse":
            current_rmse,

        "current_r2":
            current_r2,

        "registered_rmse":
            registered_rmse,

        "rmse_change":
            rmse_change,

        "rmse_change_percentage":
            rmse_change_percentage,

        "rmse_threshold":
            rmse_threshold,

        "performance_status":
            performance_status,
    }


# ============================================================
# Evaluate Retraining Need
# ============================================================

def evaluate_retraining_need(
    values,
):
    """
    Decide whether retraining is required.

    Retrain when:

        1. Current RMSE >= RMSE alert threshold

    OR

        2. Significant feature drift exists
           AND
           current RMSE has degraded by at least the
           configured percentage from registered RMSE.

    This prevents seasonal distribution shifts alone
    from unnecessarily triggering retraining.
    """

    reasons = []

    observations = []


    # ========================================================
    # Drift
    # ========================================================

    drift_detected = (
        values[
            "overall_status"
        ]
        in {
            "DRIFT_DETECTED",
            "MODERATE_DRIFT",
        }
    )


    significant_drift = (
        values[
            "drift_percentage"
        ]
        >= DRIFT_PERCENTAGE_THRESHOLD
    )


    if significant_drift:

        observations.append(
            "Significant feature drift detected: "
            f"{values['drift_percentage']:.2f}% "
            "of monitored numeric features."
        )


    # ========================================================
    # Absolute Performance Failure
    # ========================================================

    performance_threshold_exceeded = False

    if (
        values[
            "performance_available"
        ]
        and
        values[
            "current_rmse"
        ]
        is not None
    ):

        performance_threshold_exceeded = (
            values[
                "current_rmse"
            ]
            >= values[
                "rmse_threshold"
            ]
        )


    if performance_threshold_exceeded:

        reasons.append(
            "Current production RMSE exceeded "
            "the configured alert threshold: "
            f"{values['current_rmse']:.4f} "
            f">= "
            f"{values['rmse_threshold']:.4f}."
        )


    # ========================================================
    # Relative Performance Degradation
    # ========================================================

    rmse_meaningfully_degraded = False

    rmse_change_percentage = (
        values.get(
            "rmse_change_percentage"
        )
    )


    if (
        values[
            "performance_available"
        ]
        and
        rmse_change_percentage is not None
    ):

        rmse_meaningfully_degraded = (
            rmse_change_percentage
            >= RMSE_DEGRADATION_PERCENT_THRESHOLD
        )


    if rmse_meaningfully_degraded:

        observations.append(
            "Current RMSE increased by "
            f"{rmse_change_percentage:.2f}% "
            "relative to the registered production baseline."
        )


    # ========================================================
    # Drift + Degradation Combination
    # ========================================================

    drift_with_degradation = (
        significant_drift
        and
        rmse_meaningfully_degraded
    )


    if drift_with_degradation:

        reasons.append(
            "Significant feature drift is accompanied "
            "by meaningful model-performance degradation."
        )


    # ========================================================
    # Stable Performance During Drift
    # ========================================================

    seasonal_or_covariate_drift_only = (
        significant_drift
        and
        not rmse_meaningfully_degraded
        and
        not performance_threshold_exceeded
    )


    if seasonal_or_covariate_drift_only:

        observations.append(
            "Drift is currently not accompanied by "
            "meaningful RMSE degradation. "
            "Continue monitoring without retraining."
        )


    # ========================================================
    # Final Decision
    # ========================================================

    retraining_required = (
        performance_threshold_exceeded
        or
        drift_with_degradation
    )


    return {
        "retraining_required":
            retraining_required,

        "drift_detected":
            drift_detected,

        "significant_drift":
            significant_drift,

        "performance_threshold_exceeded":
            performance_threshold_exceeded,

        "rmse_meaningfully_degraded":
            rmse_meaningfully_degraded,

        "drift_with_degradation":
            drift_with_degradation,

        "seasonal_or_covariate_drift_only":
            seasonal_or_covariate_drift_only,

        "reasons":
            reasons,

        "observations":
            observations,
    }


# ============================================================
# Print Monitoring Status
# ============================================================

def print_monitoring_status(
    values,
):
    """
    Display current monitoring information.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CURRENT PRODUCTION STATUS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nProduction Model     : "
        f"{values['production_model']}"
    )

    print(
        f"Production Version   : "
        f"{values['production_version']}"
    )

    print(
        f"Overall Status       : "
        f"{values['overall_status']}"
    )


    print(
        "\nFeature Drift"
    )

    print(
        "-" * 50
    )

    print(
        f"Numeric Features     : "
        f"{values['numeric_features']}"
    )

    print(
        f"Drift Features       : "
        f"{values['drift_features']}"
    )

    print(
        f"Moderate Drift       : "
        f"{values['moderate_drift_features']}"
    )

    print(
        f"No Drift Features    : "
        f"{values['no_drift_features']}"
    )

    print(
        f"Drift Percentage     : "
        f"{values['drift_percentage']:.2f}%"
    )

    print(
        f"Drift Threshold      : "
        f"{DRIFT_PERCENTAGE_THRESHOLD:.2f}%"
    )

    if values[
        "mean_psi"
    ] is not None:

        print(
            f"Mean PSI             : "
            f"{values['mean_psi']:.6f}"
        )


    print(
        "\nModel Performance"
    )

    print(
        "-" * 50
    )

    print(
        f"Current MAE          : "
        f"{values['current_mae']}"
    )

    print(
        f"Current RMSE         : "
        f"{values['current_rmse']}"
    )

    print(
        f"Current R²           : "
        f"{values['current_r2']}"
    )

    print(
        f"Registered RMSE      : "
        f"{values['registered_rmse']}"
    )

    print(
        f"RMSE Alert Threshold : "
        f"{values['rmse_threshold']}"
    )

    print(
        f"Performance Status   : "
        f"{values['performance_status']}"
    )

    if (
        values[
            "rmse_change_percentage"
        ]
        is not None
    ):

        print(
            f"RMSE Change          : "
            f"{values['rmse_change_percentage']:.2f}%"
        )

        print(
            f"Allowed Degradation  : "
            f"{RMSE_DEGRADATION_PERCENT_THRESHOLD:.2f}%"
        )


# ============================================================
# Trigger Retraining
# ============================================================

def trigger_retraining():
    """
    Execute the existing model comparison / retraining module.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING AUTOMATED RETRAINING"
    )

    print(
        "=" * 70
    )


    print(
        f"\nExecuting:"
    )

    print(
        f"python -m {RETRAIN_MODULE}"
    )


    result = subprocess.run(
        [
            sys.executable,
            "-m",
            RETRAIN_MODULE,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )


    if result.returncode != 0:

        raise RuntimeError(
            "Automated model retraining failed.\n"
            f"Exit code: {result.returncode}"
        )


    print(
        "\nAutomated retraining "
        "completed successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "AQI AUTOMATED RETRAINING CHECK"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\nLoading latest monitoring summary..."
    )


    summary = (
        load_monitoring_summary()
    )


    print(
        "Monitoring summary loaded successfully."
    )


    # ========================================================
    # STEP 2
    # ========================================================

    values = (
        extract_monitoring_values(
            summary
        )
    )


    print_monitoring_status(
        values
    )


    # ========================================================
    # STEP 3
    # ========================================================

    decision = (
        evaluate_retraining_need(
            values
        )
    )


    # ========================================================
    # Decision Output
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRAINING DECISION"
    )

    print(
        "=" * 70
    )


    print(
        f"\nDrift Detected           : "
        f"{decision['drift_detected']}"
    )

    print(
        f"Significant Drift        : "
        f"{decision['significant_drift']}"
    )

    print(
        f"RMSE Threshold Exceeded  : "
        f"{decision['performance_threshold_exceeded']}"
    )

    print(
        f"RMSE Meaningfully Worse  : "
        f"{decision['rmse_meaningfully_degraded']}"
    )

    print(
        f"Drift + Degradation      : "
        f"{decision['drift_with_degradation']}"
    )


    print(
        "\n" + "-" * 70
    )


    # ========================================================
    # Retraining Required
    # ========================================================

    if decision[
        "retraining_required"
    ]:

        print(
            "DECISION                 : "
            "RETRAINING REQUIRED"
        )


        if decision[
            "reasons"
        ]:

            print(
                "\nReasons:"
            )

            for reason in decision[
                "reasons"
            ]:

                print(
                    f"  - {reason}"
                )


        trigger_retraining()


    # ========================================================
    # No Retraining
    # ========================================================

    else:

        print(
            "DECISION                 : "
            "NO RETRAINING REQUIRED"
        )


        if decision[
            "observations"
        ]:

            print(
                "\nMonitoring observations:"
            )

            for observation in decision[
                "observations"
            ]:

                print(
                    f"  - {observation}"
                )


        print(
            "\nProduction model remains acceptable."
        )

        print(
            "No automated retraining will be started."
        )


        if decision[
            "seasonal_or_covariate_drift_only"
        ]:

            print(
                "\nRecommendation:"
            )

            print(
                "Continue monitoring the detected "
                "distribution shift. Retrain only if "
                "future performance begins to degrade."
            )


    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRAINING CHECK COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()