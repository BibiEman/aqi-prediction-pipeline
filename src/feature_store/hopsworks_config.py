"""
hopsworks_config.py

Central Hopsworks configuration for the
AQI Prediction System.
"""

import os

import hopsworks
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Project
# ============================================================

PROJECT_NAME = "AQI_Prediction_3days"


# ============================================================
# Historical Training Feature Group
# ============================================================

TRAINING_FEATURE_GROUP_NAME = "aqi_features"
TRAINING_FEATURE_GROUP_VERSION = 1


# ============================================================
# Real-Time Online Feature Group
# ============================================================

REALTIME_FEATURE_GROUP_NAME = "aqi_realtime_features"

# v1 was created for the earlier offline attempt.
# v2 is dedicated to online/live serving.
REALTIME_FEATURE_GROUP_VERSION = 2


# ============================================================
# Connection
# ============================================================

def get_hopsworks_project():
    """
    Connect to Hopsworks.
    """

    print("\nConnecting to Hopsworks...")

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if api_key:

        project = hopsworks.login(
            project=PROJECT_NAME,
            api_key_value=api_key,
        )

    else:

        project = hopsworks.login(
            project=PROJECT_NAME,
        )

    print(
        f"Connected project: {PROJECT_NAME}"
    )

    return project


def get_feature_store():
    """
    Return Hopsworks Feature Store handle.
    """

    project = get_hopsworks_project()

    feature_store = (
        project.get_feature_store()
    )

    print(
        "Feature Store connection established."
    )

    return feature_store


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HOPSWORKS CONNECTION TEST")
    print("=" * 70)

    get_feature_store()

    print(
        "\nHopsworks connection test successful."
    )