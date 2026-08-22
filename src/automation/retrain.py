"""
retrain.py

Automated AQI retraining orchestration.

This module runs:

    Monitoring
        ↓
    Retraining decision
        ↓
    Retraining only if required

The actual retraining decision is handled by:
    src.monitoring.retraining_trigger

This script acts as a simple orchestration entry point.
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
# Run Module
# ============================================================

def run_module(
    description,
    module,
):
    """
    Run a Python module and stop if it fails.
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
            f"{description} failed.\n"
            f"Module: {module}\n"
            f"Exit code: {result.returncode}"
        )

    print(
        f"\n{description} completed successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI AUTOMATED RETRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Model Monitoring
    # --------------------------------------------------------

    run_module(
        description=(
            "STEP 1 - PRODUCTION MODEL MONITORING"
        ),
        module=(
            "src.monitoring.monitor"
        ),
    )

    # --------------------------------------------------------
    # Step 2: Retraining Decision
    # --------------------------------------------------------

    run_module(
        description=(
            "STEP 2 - RETRAINING DECISION"
        ),
        module=(
            "src.monitoring.retraining_trigger"
        ),
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUTOMATED RETRAINING CHECK COMPLETED")
    print("=" * 70)

    print(
        "\nIf model performance was stable, "
        "no retraining was performed."
    )

    print(
        "If actionable degradation was detected, "
        "the retraining trigger handled the retraining process."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()