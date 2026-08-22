"""
hourly_pipeline.py

Hourly production feature pipeline for the
AQI Prediction System.

Workflow
--------
OpenWeather APIs
        ↓
Collect current weather + pollutants
        ↓
Append raw observation history
        ↓
Build model-compatible 3-hour features
        ↓
Upload latest city features
        ↓
Hopsworks Online Feature Store
"""

import subprocess
import sys

from datetime import (
    datetime,
    timezone,
)

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
# Run Pipeline Module
# ============================================================

def run_module(
    description,
    module,
):
    """
    Execute one Python module.

    Raises
    ------
    RuntimeError
        If the module exits with a non-zero status.
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

        print(
            "\n" + "!" * 70
        )

        print(
            f"FAILED: {description}"
        )

        print(
            "!" * 70
        )

        raise RuntimeError(
            f"{description} failed.\n"
            f"Module: {module}\n"
            f"Exit code: {result.returncode}"
        )

    print(
        f"\n{description} "
        "completed successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    start_time = datetime.now(
        timezone.utc
    )

    print("\n" + "=" * 70)
    print("AQI HOURLY PRODUCTION FEATURE PIPELINE")
    print("=" * 70)

    print(
        f"\nStarted at: "
        f"{start_time.isoformat()}"
    )

    # ========================================================
    # STEP 1
    # Collect new API observations
    # ========================================================

    run_module(
        description=(
            "STEP 1 - COLLECT CURRENT AQI "
            "AND WEATHER DATA"
        ),
        module=(
            "src.data_collection.collect_data"
        ),
    )

    # ========================================================
    # STEP 2
    # Build three-hour model features
    # ========================================================

    run_module(
        description=(
            "STEP 2 - BUILD 3-HOUR "
            "REAL-TIME FEATURES"
        ),
        module=(
            "src.realtime.realtime_features"
        ),
    )

    # ========================================================
    # STEP 3
    # Upload latest feature vectors
    # ========================================================

    run_module(
        description=(
            "STEP 3 - UPDATE HOPSWORKS "
            "ONLINE FEATURE STORE"
        ),
        module=(
            "src.feature_store.feature_group"
        ),
    )

    # ========================================================
    # Complete
    # ========================================================

    end_time = datetime.now(
        timezone.utc
    )

    duration = (
        end_time
        - start_time
    ).total_seconds()

    print("\n" + "=" * 70)
    print("HOURLY PRODUCTION PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nStarted   : "
        f"{start_time.isoformat()}"
    )

    print(
        f"Completed : "
        f"{end_time.isoformat()}"
    )

    print(
        f"Duration  : "
        f"{duration:.2f} seconds"
    )

    print(
        "\nCompleted stages:"
    )

    print(
        "  ✓ Live data collection"
    )

    print(
        "  ✓ 3-hour feature engineering"
    )

    print(
        "  ✓ Hopsworks online feature ingestion"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()