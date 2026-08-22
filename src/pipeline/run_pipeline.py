"""
run_pipeline.py

Complete manual AQI machine-learning pipeline.

This script is intended for a FULL pipeline rebuild.

Workflow
--------
Feature Engineering
        ↓
Prepare Training Dataset
        ↓
Train Models
        ↓
Evaluate Models
        ↓
Register Best Model
        ↓
SHAP Explainability
        ↓
Production Monitoring
"""

import subprocess
import sys
from pathlib import Path


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Run Step
# ============================================================

def run_step(
    description,
    module,
):
    """
    Execute one pipeline module.
    """

    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    print(
        f"\nRunning:\n"
        f"python -m {module}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"\nPipeline stopped.\n"
            f"Failed step: {description}\n"
            f"Module: {module}\n"
            f"Exit code: {result.returncode}"
        )

    print(
        f"\n{description} completed successfully."
    )


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 1 - FEATURE ENGINEERING"
        ),
        module=(
            "src.feature_engineering.build_features"
        ),
    )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 2 - PREPARE TRAINING DATASET"
        ),
        module=(
            "src.feature_engineering.prepare_training"
        ),
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 3 - TRAIN MACHINE LEARNING MODELS"
        ),
        module=(
            "src.model_training.train_models"
        ),
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 4 - EVALUATE MACHINE LEARNING MODELS"
        ),
        module=(
            "src.model_training.evaluate_models"
        ),
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 5 - REGISTER BEST MODEL"
        ),
        module=(
            "src.model_training.register_best_model"
        ),
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 6 - SHAP MODEL EXPLAINABILITY"
        ),
        module=(
            "src.explainability.shap_explainer"
        ),
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    run_step(
        description=(
            "STEP 7 - PRODUCTION MONITORING"
        ),
        module=(
            "src.monitoring.monitor"
        ),
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AQI FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nCompleted stages:"
    )

    print(
        "  ✓ Feature engineering"
    )

    print(
        "  ✓ Training dataset preparation"
    )

    print(
        "  ✓ Model training"
    )

    print(
        "  ✓ Model evaluation"
    )

    print(
        "  ✓ Best-model registration"
    )

    print(
        "  ✓ SHAP explainability"
    )

    print(
        "  ✓ Production monitoring"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()