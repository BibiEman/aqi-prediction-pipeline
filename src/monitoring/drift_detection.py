"""
drift_detection.py

Reusable data-drift detection utilities for the
AQI Prediction Pipeline.

Responsibilities
----------------
- Calculate Population Stability Index (PSI)
- Detect numerical feature drift
- Detect categorical feature drift
- Classify drift severity
- Summarize drift statistics

This module does not:
- load the model registry
- retrain models
- save monitoring history

Those responsibilities belong to monitor.py and
retraining_trigger.py.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================
# Drift Thresholds
# ============================================================

PSI_NO_DRIFT = 0.10

PSI_MODERATE_DRIFT = 0.25

CATEGORICAL_NO_DRIFT = 0.10

CATEGORICAL_MODERATE_DRIFT = 0.25

EPSILON = 1e-6


# ============================================================
# Columns Excluded From Numerical Drift
# ============================================================

EXCLUDED_COLUMNS = {
    "timestamp",
    "target_aqi",
}


# ============================================================
# Categorical Features
# ============================================================

DEFAULT_CATEGORICAL_FEATURES = [
    "city",
    "day_of_week",
    "season",
]


# ============================================================
# PSI Calculation
# ============================================================

def calculate_psi(
    reference,
    current,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI interpretation
    ------------------
    PSI < 0.10
        No meaningful drift.

    0.10 <= PSI < 0.25
        Moderate drift.

    PSI >= 0.25
        Significant drift.

    Parameters
    ----------
    reference
        Reference/training feature values.

    current
        Current/recent feature values.

    bins : int
        Number of quantile-based bins.

    Returns
    -------
    float
        PSI score.
    """

    reference = pd.Series(
        reference
    ).dropna()

    current = pd.Series(
        current
    ).dropna()

    # --------------------------------------------------------
    # Empty data
    # --------------------------------------------------------

    if reference.empty:
        return np.nan

    if current.empty:
        return np.nan

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if reference.empty:
        return np.nan

    if current.empty:
        return np.nan

    # --------------------------------------------------------
    # Constant feature handling
    # --------------------------------------------------------

    if reference.nunique() <= 1:

        reference_value = (
            float(reference.iloc[0])
        )

        # Current also constant and same
        if (
            current.nunique() <= 1
            and float(current.iloc[0])
            == reference_value
        ):

            return 0.0

        # Distribution changed from a constant
        return float("inf")

    # --------------------------------------------------------
    # Reference quantile bins
    # --------------------------------------------------------

    try:

        quantiles = np.linspace(
            0.0,
            1.0,
            bins + 1,
        )

        edges = np.unique(
            reference.quantile(
                quantiles
            ).to_numpy()
        )

        if len(edges) < 2:
            return 0.0

        # ----------------------------------------------------
        # Extend outer limits
        # ----------------------------------------------------

        edges = edges.astype(
            float
        )

        edges[0] = -np.inf

        edges[-1] = np.inf

        # ----------------------------------------------------
        # Bin values
        # ----------------------------------------------------

        reference_bins = pd.cut(
            reference,
            bins=edges,
            include_lowest=True,
        )

        current_bins = pd.cut(
            current,
            bins=edges,
            include_lowest=True,
        )

        reference_counts = (
            reference_bins
            .value_counts(
                sort=False
            )
        )

        current_counts = (
            current_bins
            .value_counts(
                sort=False
            )
        )

        # Reindex current to ensure same bins
        current_counts = (
            current_counts.reindex(
                reference_counts.index,
                fill_value=0,
            )
        )

    except Exception:

        return np.nan

    # --------------------------------------------------------
    # Convert counts to proportions
    # --------------------------------------------------------

    reference_total = max(
        reference_counts.sum(),
        1,
    )

    current_total = max(
        current_counts.sum(),
        1,
    )

    reference_pct = (
        reference_counts
        / reference_total
    ).astype(float)

    current_pct = (
        current_counts
        / current_total
    ).astype(float)

    # --------------------------------------------------------
    # Prevent divide-by-zero
    # --------------------------------------------------------

    reference_pct = (
        reference_pct
        .clip(
            lower=EPSILON
        )
    )

    current_pct = (
        current_pct
        .clip(
            lower=EPSILON
        )
    )

    # --------------------------------------------------------
    # PSI
    # --------------------------------------------------------

    psi_values = (
        (current_pct - reference_pct)
        * np.log(
            current_pct
            / reference_pct
        )
    )

    psi = float(
        psi_values.sum()
    )

    return psi


# ============================================================
# PSI Classification
# ============================================================

def classify_psi(
    psi: float,
) -> str:
    """
    Classify PSI score.
    """

    if pd.isna(psi):
        return "UNAVAILABLE"

    if np.isinf(psi):
        return "DRIFT"

    if psi < PSI_NO_DRIFT:
        return "NO_DRIFT"

    if psi < PSI_MODERATE_DRIFT:
        return "MODERATE_DRIFT"

    return "DRIFT"


# Backward-compatible function name
def classify_drift(
    psi: float,
) -> str:
    """
    Alias for classify_psi().
    """

    return classify_psi(
        psi
    )


# ============================================================
# Numerical Feature Selection
# ============================================================

def get_numeric_features(
    dataframe: pd.DataFrame,
    excluded_columns: Optional[Iterable[str]] = None,
):
    """
    Return numerical features suitable for PSI.
    """

    if excluded_columns is None:

        excluded_columns = (
            EXCLUDED_COLUMNS
        )

    excluded_columns = set(
        excluded_columns
    )

    numeric_columns = (
        dataframe
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    return [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]


# Backward-compatible name
def get_monitoring_features(
    dataframe,
):
    """
    Alias used by monitor.py.
    """

    return get_numeric_features(
        dataframe
    )


# ============================================================
# Numerical Drift Detection
# ============================================================

def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features=None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate PSI for numerical features.

    Parameters
    ----------
    reference : pandas.DataFrame
        Historical/reference data.

    current : pandas.DataFrame
        Recent/current data.

    features : list, optional
        Features to check.

    bins : int
        PSI bin count.

    Returns
    -------
    pandas.DataFrame
        Feature-level drift report.
    """

    if features is None:

        features = get_numeric_features(
            reference
        )

    results = []

    for feature in features:

        if feature not in reference.columns:
            continue

        if feature not in current.columns:
            continue

        reference_values = (
            reference[
                feature
            ]
        )

        current_values = (
            current[
                feature
            ]
        )

        psi = calculate_psi(
            reference=reference_values,
            current=current_values,
            bins=bins,
        )

        status = classify_psi(
            psi
        )

        results.append(
            {
                "Feature": feature,

                "PSI": psi,

                "Status": status,

                "Reference Mean":
                    (
                        float(
                            pd.to_numeric(
                                reference_values,
                                errors="coerce",
                            ).mean()
                        )
                        if reference_values.notna().any()
                        else np.nan
                    ),

                "Current Mean":
                    (
                        float(
                            pd.to_numeric(
                                current_values,
                                errors="coerce",
                            ).mean()
                        )
                        if current_values.notna().any()
                        else np.nan
                    ),

                "Reference Samples":
                    int(
                        reference_values
                        .notna()
                        .sum()
                    ),

                "Current Samples":
                    int(
                        current_values
                        .notna()
                        .sum()
                    ),
            }
        )

    report = pd.DataFrame(
        results
    )

    if report.empty:
        return report

    report = (
        report
        .sort_values(
            "PSI",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return report


# ============================================================
# Total Variation Distance
# ============================================================

def calculate_distribution_distance(
    reference,
    current,
) -> float:
    """
    Calculate Total Variation Distance between
    two categorical distributions.

    Range
    -----
    0
        Identical distributions.

    1
        Completely different distributions.
    """

    reference = (
        pd.Series(
            reference
        )
        .dropna()
        .astype(str)
    )

    current = (
        pd.Series(
            current
        )
        .dropna()
        .astype(str)
    )

    if reference.empty:
        return np.nan

    if current.empty:
        return np.nan

    reference_distribution = (
        reference
        .value_counts(
            normalize=True
        )
    )

    current_distribution = (
        current
        .value_counts(
            normalize=True
        )
    )

    categories = (
        set(
            reference_distribution.index
        )
        |
        set(
            current_distribution.index
        )
    )

    distance = 0.0

    for category in categories:

        reference_value = (
            reference_distribution.get(
                category,
                0.0,
            )
        )

        current_value = (
            current_distribution.get(
                category,
                0.0,
            )
        )

        distance += abs(
            reference_value
            - current_value
        )

    return float(
        distance / 2.0
    )


# ============================================================
# Categorical Drift Classification
# ============================================================

def classify_categorical_drift(
    distance: float,
) -> str:
    """
    Classify categorical distribution distance.
    """

    if pd.isna(distance):
        return "UNAVAILABLE"

    if distance < CATEGORICAL_NO_DRIFT:
        return "NO_DRIFT"

    if (
        distance
        < CATEGORICAL_MODERATE_DRIFT
    ):
        return "MODERATE_DRIFT"

    return "DRIFT"


# ============================================================
# Categorical Drift Detection
# ============================================================

def calculate_categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical_features=None,
) -> pd.DataFrame:
    """
    Detect drift in categorical model features.
    """

    if categorical_features is None:

        categorical_features = (
            DEFAULT_CATEGORICAL_FEATURES
        )

    results = []

    for feature in categorical_features:

        if feature not in reference.columns:
            continue

        if feature not in current.columns:
            continue

        distance = (
            calculate_distribution_distance(
                reference[
                    feature
                ],
                current[
                    feature
                ],
            )
        )

        status = (
            classify_categorical_drift(
                distance
            )
        )

        results.append(
            {
                "Feature": feature,

                "Metric":
                    "Total Variation Distance",

                "Value":
                    distance,

                "Status":
                    status,

                "Reference Unique Values":
                    int(
                        reference[
                            feature
                        ].nunique(
                            dropna=True
                        )
                    ),

                "Current Unique Values":
                    int(
                        current[
                            feature
                        ].nunique(
                            dropna=True
                        )
                    ),

                "Reference Samples":
                    int(
                        reference[
                            feature
                        ]
                        .notna()
                        .sum()
                    ),

                "Current Samples":
                    int(
                        current[
                            feature
                        ]
                        .notna()
                        .sum()
                    ),
            }
        )

    report = pd.DataFrame(
        results
    )

    if report.empty:
        return report

    return (
        report
        .sort_values(
            "Value",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Drift Summary
# ============================================================

def summarize_numeric_drift(
    drift_report: pd.DataFrame,
) -> dict:
    """
    Summarize numerical feature drift.
    """

    if drift_report.empty:

        return {
            "numeric_features": 0,
            "drift_features": 0,
            "moderate_drift_features": 0,
            "no_drift_features": 0,
            "unavailable_features": 0,
            "drift_percentage": 0.0,
            "moderate_drift_percentage": 0.0,
        }

    total = len(
        drift_report
    )

    drift = int(
        (
            drift_report[
                "Status"
            ]
            == "DRIFT"
        ).sum()
    )

    moderate = int(
        (
            drift_report[
                "Status"
            ]
            == "MODERATE_DRIFT"
        ).sum()
    )

    no_drift = int(
        (
            drift_report[
                "Status"
            ]
            == "NO_DRIFT"
        ).sum()
    )

    unavailable = int(
        (
            drift_report[
                "Status"
            ]
            == "UNAVAILABLE"
        ).sum()
    )

    return {
        "numeric_features":
            total,

        "drift_features":
            drift,

        "moderate_drift_features":
            moderate,

        "no_drift_features":
            no_drift,

        "unavailable_features":
            unavailable,

        "drift_percentage":
            round(
                (
                    drift
                    / total
                    * 100
                ),
                2,
            ),

        "moderate_drift_percentage":
            round(
                (
                    moderate
                    / total
                    * 100
                ),
                2,
            ),
    }


def summarize_categorical_drift(
    categorical_report: pd.DataFrame,
) -> dict:
    """
    Summarize categorical drift report.
    """

    if categorical_report.empty:

        return {
            "features_checked": 0,
            "drift_features": 0,
            "moderate_drift_features": 0,
            "no_drift_features": 0,
        }

    return {
        "features_checked":
            len(
                categorical_report
            ),

        "drift_features":
            int(
                (
                    categorical_report[
                        "Status"
                    ]
                    == "DRIFT"
                ).sum()
            ),

        "moderate_drift_features":
            int(
                (
                    categorical_report[
                        "Status"
                    ]
                    == "MODERATE_DRIFT"
                ).sum()
            ),

        "no_drift_features":
            int(
                (
                    categorical_report[
                        "Status"
                    ]
                    == "NO_DRIFT"
                ).sum()
            ),
    }


# ============================================================
# Complete Drift Analysis
# ============================================================

def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Execute numerical and categorical drift detection.

    Returns
    -------
    dict
        Complete drift analysis.
    """

    numeric_report = (
        detect_feature_drift(
            reference=reference,
            current=current,
        )
    )

    categorical_report = (
        calculate_categorical_drift(
            reference=reference,
            current=current,
        )
    )

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

    return {
        "numeric_report":
            numeric_report,

        "categorical_report":
            categorical_report,

        "numeric_summary":
            numeric_summary,

        "categorical_summary":
            categorical_summary,
    }


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AQI DRIFT DETECTION MODULE")
    print("=" * 70)

    print(
        "\nThis module contains reusable drift "
        "detection functions."
    )

    print(
        "\nRun the complete monitoring pipeline using:"
    )

    print(
        "\npython -m src.monitoring.monitor"
    )