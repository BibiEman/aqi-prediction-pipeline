"""
shap_explainer.py

SHAP explainability pipeline for the AQI Prediction System.

Purpose
-------
Explain the production/best LightGBM AQI model using SHAP.

This script:
    1. Loads the trained LightGBM pipeline.
    2. Loads the saved chronological evaluation test set.
    3. Applies the pipeline's trained preprocessing.
    4. Extracts transformed feature names.
    5. Calculates SHAP values using TreeExplainer.
    6. Generates global feature-importance plots.
    7. Saves feature importance values to CSV.
    8. Generates an example local prediction explanation.

Important
---------
The saved LightGBM.joblib file contains:

    Raw DataFrame
          ↓
    ColumnTransformer
          ↓
    OneHotEncoder / FunctionTransformer
          ↓
    LightGBM

SHAP therefore explains the transformed input received by
the LightGBM estimator, not the raw DataFrame directly.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib

# Use a non-interactive backend so plots save cleanly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# Configuration
# ============================================================

warnings.filterwarnings(
    "ignore"
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


EXPLAINABILITY_DIR = (
    RESULTS_DIR
    / "explainability"
)


EXPLAINABILITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_NAME = "LightGBM"


MODEL_PATH = (
    MODEL_DIR
    / f"{MODEL_NAME}.joblib"
)


EVALUATION_DATA_PATH = (
    MODEL_DIR
    / "evaluation_test_data.joblib"
)


FEATURE_IMPORTANCE_CSV = (
    EXPLAINABILITY_DIR
    / "shap_feature_importance.csv"
)


SUMMARY_PLOT_PATH = (
    EXPLAINABILITY_DIR
    / "shap_summary_plot.png"
)


BAR_PLOT_PATH = (
    EXPLAINABILITY_DIR
    / "shap_feature_importance.png"
)


LOCAL_EXPLANATION_PATH = (
    EXPLAINABILITY_DIR
    / "shap_local_explanation.png"
)


LOCAL_EXPLANATION_CSV = (
    EXPLAINABILITY_DIR
    / "shap_local_explanation.csv"
)


# ============================================================
# SHAP Configuration
# ============================================================

# SHAP can be expensive on very large datasets.
# 2,000 samples is more than enough for a good project-level
# global explanation.

SHAP_SAMPLE_SIZE = 2000

RANDOM_STATE = 42


# ============================================================
# Load Model
# ============================================================

def load_lightgbm_pipeline():
    """
    Load the trained LightGBM preprocessing + model pipeline.
    """

    print("\n" + "=" * 70)
    print("LOADING LIGHTGBM MODEL")
    print("=" * 70)

    print(
        f"\nModel path:\n"
        f"{MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nLightGBM model was not found.\n"
            f"Expected:\n{MODEL_PATH}\n\n"
            "Run the model training pipeline first."
        )

    pipeline = joblib.load(
        MODEL_PATH
    )

    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise TypeError(
            "Loaded LightGBM artifact is not "
            "an sklearn Pipeline."
        )

    required_steps = [
        "preprocessor",
        "model",
    ]

    missing_steps = [
        step
        for step in required_steps
        if step not in pipeline.named_steps
    ]

    if missing_steps:

        raise ValueError(
            "Required pipeline steps are missing:\n"
            f"{missing_steps}"
        )

    print(
        "\nLightGBM pipeline loaded successfully."
    )

    return pipeline


# ============================================================
# Load Evaluation Data
# ============================================================

def load_evaluation_data():
    """
    Load the chronological test split saved during training.
    """

    print("\n" + "=" * 70)
    print("LOADING EVALUATION DATA")
    print("=" * 70)

    print(
        f"\nEvaluation data:\n"
        f"{EVALUATION_DATA_PATH}"
    )

    if not EVALUATION_DATA_PATH.exists():

        raise FileNotFoundError(
            "\nEvaluation test data was not found.\n"
            f"Expected:\n{EVALUATION_DATA_PATH}"
        )

    evaluation_data = joblib.load(
        EVALUATION_DATA_PATH
    )

    if not isinstance(
        evaluation_data,
        dict,
    ):

        raise TypeError(
            "Evaluation data must be stored as a dictionary."
        )

    if "X_test" not in evaluation_data:

        raise KeyError(
            "X_test was not found in evaluation data."
        )

    if "y_test" not in evaluation_data:

        raise KeyError(
            "y_test was not found in evaluation data."
        )

    X_test = evaluation_data[
        "X_test"
    ]

    y_test = evaluation_data[
        "y_test"
    ]

    if not isinstance(
        X_test,
        pd.DataFrame,
    ):

        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    print(
        f"\nX_test shape : {X_test.shape}"
    )

    print(
        f"y_test shape : {y_test.shape}"
    )

    return (
        X_test,
        y_test,
    )


# ============================================================
# Sample Evaluation Data
# ============================================================

def sample_test_data(
    X_test,
):
    """
    Sample test observations for SHAP analysis.
    """

    print("\n" + "=" * 70)
    print("SAMPLING DATA FOR SHAP")
    print("=" * 70)

    if len(X_test) <= SHAP_SAMPLE_SIZE:

        sample = (
            X_test.copy()
            .reset_index(
                drop=True
            )
        )

    else:

        sample = (
            X_test.sample(
                n=SHAP_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
            .sort_index()
            .reset_index(
                drop=True
            )
        )

    print(
        f"\nOriginal test rows : "
        f"{len(X_test):,}"
    )

    print(
        f"SHAP sample rows   : "
        f"{len(sample):,}"
    )

    return sample


# ============================================================
# Extract Transformed Feature Names
# ============================================================

def get_transformed_feature_names(
    preprocessor,
):
    """
    Recover feature names after ColumnTransformer processing.

    Handles:
        - OneHotEncoder categorical features
        - explicit numerical FunctionTransformer features
    """

    print("\n" + "=" * 70)
    print("EXTRACTING TRANSFORMED FEATURE NAMES")
    print("=" * 70)

    feature_names = []

    # transformers_ is available after pipeline.fit()
    transformers = getattr(
        preprocessor,
        "transformers_",
        None,
    )

    if transformers is None:

        raise RuntimeError(
            "Preprocessor does not appear to be fitted."
        )

    for (
        transformer_name,
        transformer,
        columns,
    ) in transformers:

        # ----------------------------------------------------
        # Ignore dropped remainder
        # ----------------------------------------------------

        if transformer_name == "remainder":
            continue

        if transformer == "drop":
            continue

        # ----------------------------------------------------
        # Normalize columns
        # ----------------------------------------------------

        if isinstance(
            columns,
            str,
        ):

            columns = [
                columns
            ]

        else:

            try:

                columns = list(
                    columns
                )

            except TypeError:

                continue

        # ----------------------------------------------------
        # Categorical transformer
        # ----------------------------------------------------

        if transformer_name == "categorical":

            if hasattr(
                transformer,
                "get_feature_names_out",
            ):

                encoded_names = (
                    transformer
                    .get_feature_names_out(
                        columns
                    )
                )

                feature_names.extend(
                    [
                        str(name)
                        for name
                        in encoded_names
                    ]
                )

            else:

                feature_names.extend(
                    [
                        str(column)
                        for column
                        in columns
                    ]
                )

        # ----------------------------------------------------
        # Numerical transformer
        # ----------------------------------------------------

        elif transformer_name == "numerical":

            feature_names.extend(
                [
                    str(column)
                    for column
                    in columns
                ]
            )

        # ----------------------------------------------------
        # Other transformers
        # ----------------------------------------------------

        else:

            if hasattr(
                transformer,
                "get_feature_names_out",
            ):

                try:

                    names = (
                        transformer
                        .get_feature_names_out(
                            columns
                        )
                    )

                    feature_names.extend(
                        [
                            str(name)
                            for name
                            in names
                        ]
                    )

                except Exception:

                    feature_names.extend(
                        [
                            str(column)
                            for column
                            in columns
                        ]
                    )

            else:

                feature_names.extend(
                    [
                        str(column)
                        for column
                        in columns
                    ]
                )

    print(
        f"\nTransformed feature count: "
        f"{len(feature_names)}"
    )

    print(
        "\nFirst transformed features:"
    )

    for name in feature_names[:20]:

        print(
            f"  - {name}"
        )

    return feature_names


# ============================================================
# Transform SHAP Input
# ============================================================

def transform_features(
    pipeline,
    X_sample,
):
    """
    Apply the already-fitted preprocessing pipeline.
    """

    print("\n" + "=" * 70)
    print("TRANSFORMING SHAP INPUT")
    print("=" * 70)

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    transformed = (
        preprocessor.transform(
            X_sample
        )
    )

    # Convert sparse matrix if necessary.
    if hasattr(
        transformed,
        "toarray",
    ):

        transformed = (
            transformed.toarray()
        )

    transformed = np.asarray(
        transformed
    )

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if (
        transformed.shape[1]
        != len(feature_names)
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Transformed feature count and extracted "
            "feature-name count do not match."
        )

        print(
            f"Matrix columns : "
            f"{transformed.shape[1]}"
        )

        print(
            f"Feature names  : "
            f"{len(feature_names)}"
        )

        # Safe fallback names
        feature_names = [
            f"feature_{index}"
            for index
            in range(
                transformed.shape[1]
            )
        ]

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    print(
        f"\nTransformed SHAP matrix: "
        f"{transformed_df.shape}"
    )

    return transformed_df


# ============================================================
# Calculate SHAP Values
# ============================================================

def calculate_shap_values(
    pipeline,
    transformed_df,
):
    """
    Calculate SHAP values for the trained LightGBM model.
    """

    print("\n" + "=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    print(
        f"\nEstimator type: "
        f"{type(model).__name__}"
    )

    # TreeExplainer is optimized for LightGBM.
    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        transformed_df
    )

    print(
        "\nSHAP calculation completed."
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.values.shape}"
    )

    return (
        explainer,
        shap_values,
    )


# ============================================================
# Global Feature Importance
# ============================================================

def create_feature_importance(
    shap_values,
    feature_names,
):
    """
    Calculate mean absolute SHAP importance.
    """

    print("\n" + "=" * 70)
    print("CALCULATING GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)

    values = np.asarray(
        shap_values.values
    )

    # For regression this should normally be:
    # samples x features.
    if values.ndim != 2:

        raise RuntimeError(
            "Unexpected SHAP value dimensions: "
            f"{values.shape}"
        )

    mean_absolute_shap = (
        np.abs(
            values
        )
        .mean(
            axis=0
        )
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_Absolute_SHAP": (
                mean_absolute_shap
            ),
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df.insert(
        0,
        "Rank",
        range(
            1,
            len(importance_df) + 1,
        ),
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_CSV,
        index=False,
    )

    print(
        "\nTop 20 important features:"
    )

    print(
        importance_df
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nFeature importance saved to:"
    )

    print(
        FEATURE_IMPORTANCE_CSV
    )

    return importance_df


# ============================================================
# SHAP Summary Plot
# ============================================================

def save_summary_plot(
    shap_values,
):
    """
    Save SHAP beeswarm summary plot.
    """

    print("\nGenerating SHAP summary plot...")

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.title(
        "AQI Model SHAP Summary",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        SUMMARY_PLOT_PATH
    )


# ============================================================
# SHAP Bar Plot
# ============================================================

def save_bar_plot(
    shap_values,
):
    """
    Save SHAP global feature-importance bar chart.
    """

    print(
        "\nGenerating SHAP importance bar plot..."
    )

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.title(
        "Top AQI Prediction Features",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        BAR_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:"
    )

    print(
        BAR_PLOT_PATH
    )


# ============================================================
# Local Explanation
# ============================================================

def save_local_explanation(
    pipeline,
    X_sample,
    transformed_df,
    shap_values,
):
    """
    Create a local explanation for one AQI prediction.
    """

    print("\n" + "=" * 70)
    print("GENERATING LOCAL EXPLANATION")
    print("=" * 70)

    # First sampled observation
    sample_index = 0

    raw_sample = X_sample.iloc[
        [
            sample_index
        ]
    ]

    prediction = float(
        pipeline.predict(
            raw_sample
        )[0]
    )

    local_values = (
        shap_values[
            sample_index
        ]
    )

    print(
        f"\nExample predicted AQI: "
        f"{prediction:.2f}"
    )

    # --------------------------------------------------------
    # Waterfall plot
    # --------------------------------------------------------

    plt.figure()

    shap.plots.waterfall(
        local_values,
        max_display=15,
        show=False,
    )

    plt.title(
        f"Local AQI Explanation "
        f"(Prediction = {prediction:.2f})",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        LOCAL_EXPLANATION_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Local contribution CSV
    # --------------------------------------------------------

    contribution_df = pd.DataFrame(
        {
            "Feature": (
                transformed_df.columns
            ),

            "Feature_Value": (
                transformed_df
                .iloc[
                    sample_index
                ]
                .values
            ),

            "SHAP_Value": (
                np.asarray(
                    local_values.values
                )
            ),
        }
    )

    contribution_df[
        "Absolute_SHAP"
    ] = (
        contribution_df[
            "SHAP_Value"
        ].abs()
    )

    contribution_df = (
        contribution_df
        .sort_values(
            "Absolute_SHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    contribution_df[
        "Predicted_AQI"
    ] = prediction

    contribution_df.to_csv(
        LOCAL_EXPLANATION_CSV,
        index=False,
    )

    print(
        "\nLocal explanation saved:"
    )

    print(
        LOCAL_EXPLANATION_PATH
    )

    print(
        LOCAL_EXPLANATION_CSV
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    Run the complete SHAP explainability pipeline.
    """

    print("\n" + "=" * 70)
    print("AQI MODEL EXPLAINABILITY PIPELINE")
    print("=" * 70)

    print(
        f"\nModel selected : "
        f"{MODEL_NAME}"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 1: LOADING TRAINED MODEL")
    print("-" * 70)

    pipeline = (
        load_lightgbm_pipeline()
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 2: LOADING TEST DATA")
    print("-" * 70)

    (
        X_test,
        y_test,
    ) = load_evaluation_data()

    # ========================================================
    # STEP 3
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 3: SAMPLING TEST DATA")
    print("-" * 70)

    X_sample = sample_test_data(
        X_test
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 4: APPLYING TRAINED PREPROCESSING")
    print("-" * 70)

    transformed_df = (
        transform_features(
            pipeline,
            X_sample,
        )
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 5: COMPUTING SHAP VALUES")
    print("-" * 70)

    (
        explainer,
        shap_values,
    ) = calculate_shap_values(
        pipeline,
        transformed_df,
    )

    # ========================================================
    # STEP 6
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 6: GLOBAL FEATURE IMPORTANCE")
    print("-" * 70)

    importance_df = (
        create_feature_importance(
            shap_values,
            transformed_df.columns.tolist(),
        )
    )

    # ========================================================
    # STEP 7
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 7: GENERATING GLOBAL PLOTS")
    print("-" * 70)

    save_summary_plot(
        shap_values
    )

    save_bar_plot(
        shap_values
    )

    # ========================================================
    # STEP 8
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 8: GENERATING LOCAL EXPLANATION")
    print("-" * 70)

    save_local_explanation(
        pipeline=pipeline,
        X_sample=X_sample,
        transformed_df=transformed_df,
        shap_values=shap_values,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nSamples explained : "
        f"{len(X_sample):,}"
    )

    print(
        f"Features explained: "
        f"{transformed_df.shape[1]}"
    )

    print(
        "\nTop 10 model features:"
    )

    print(
        importance_df
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  ✓ {FEATURE_IMPORTANCE_CSV}"
    )

    print(
        f"  ✓ {SUMMARY_PLOT_PATH}"
    )

    print(
        f"  ✓ {BAR_PLOT_PATH}"
    )

    print(
        f"  ✓ {LOCAL_EXPLANATION_PATH}"
    )

    print(
        f"  ✓ {LOCAL_EXPLANATION_CSV}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()