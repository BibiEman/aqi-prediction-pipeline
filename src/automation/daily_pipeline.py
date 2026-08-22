"""
daily_pipeline.py

Daily automation pipeline for the AQI Prediction System.

Purpose
-------
Run production monitoring once per day and determine
whether retraining is necessary.

Workflow
--------
Production Monitoring
        ↓
Drift Detection
        ↓
Performance Monitoring
        ↓
Retraining Trigger
        ↓
Retrain only if required
"""

import subprocess
import sys
from datetime import datetime, timezone
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

    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    print(
        f"\nCommand:\n"
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
            f"Exit code: {result.returncode}"
        )

    print(
        f"\n{description} completed."
    )


# ============================================================
# Main
# ============================================================

def main():

    start_time = datetime.now(
        timezone.utc
    )

    print("\n" + "=" * 70)
    print("AQI DAILY MLOPS PIPELINE")
    print("=" * 70)

    print(
        f"\nStarted at: "
        f"{start_time.isoformat()}"
    )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    run_module(
        description=(
            "STEP 1 - MODEL AND DATA MONITORING"
        ),
        module=(
            "src.monitoring.monitor"
        ),
    )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    run_module(
        description=(
            "STEP 2 - RETRAINING ASSESSMENT"
        ),
        module=(
            "src.monitoring.retraining_trigger"
        ),
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    end_time = datetime.now(
        timezone.utc
    )

    duration = (
        end_time
        - start_time
    ).total_seconds()

    print("\n" + "=" * 70)
    print("DAILY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nCompleted at : "
        f"{end_time.isoformat()}"
    )

    print(
        f"Duration     : "
        f"{duration:.2f} seconds"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()