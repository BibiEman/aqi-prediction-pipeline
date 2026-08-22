"""
config.py

Central configuration file for the AQI Prediction Pipeline.
"""

from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"

PLOTS_DIR = RESULTS_DIR / "plots"

REPORTS_DIR = RESULTS_DIR / "reports"

# Create folders if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Dataset
# =====================================================

DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "model_training_dataset.csv"
)

TARGET_COLUMN = "target_aqi"

# =====================================================
# Features
# =====================================================

DROP_COLUMNS = [
    "timestamp",
    TARGET_COLUMN
]

CATEGORICAL_FEATURES = [
    "city",
    "day_of_week",
    "season"
]

# =====================================================
# Train/Test Split
# =====================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =====================================================
# Models To Train
# =====================================================

MODELS = {

    "RandomForest": True,

    "ExtraTrees": True,

    "GradientBoosting": True,

    "XGBoost": True,

    "LightGBM": True,

    "CatBoost": True

}

# =====================================================
# Random Forest Parameters
# =====================================================

RF_PARAMS = {

    "n_estimators": 300,

    "max_depth": 20,

    "random_state": RANDOM_STATE,

    "n_jobs": -1

}

# =====================================================
# Extra Trees Parameters
# =====================================================

ET_PARAMS = {

    "n_estimators": 300,

    "max_depth": 20,

    "random_state": RANDOM_STATE,

    "n_jobs": -1

}

# =====================================================
# Gradient Boosting Parameters
# =====================================================

GB_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "max_depth": 5,

    "random_state": RANDOM_STATE

}

# =====================================================
# XGBoost Parameters
# =====================================================

XGB_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "max_depth": 6,

    "random_state": RANDOM_STATE,

    "n_jobs": -1

}

# =====================================================
# LightGBM Parameters
# =====================================================

LGBM_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "random_state": RANDOM_STATE

}

# =====================================================
# CatBoost Parameters
# =====================================================

CAT_PARAMS = {

    "iterations": 300,

    "learning_rate": 0.05,

    "depth": 6,

    "random_state": RANDOM_STATE,

    "verbose": 0

}

# =====================================================
# Model Registry
# =====================================================

REGISTRY_FILE = MODEL_DIR / "registry.json"

# =====================================================
# Monitoring
# =====================================================

DRIFT_THRESHOLD = 0.15

RMSE_ALERT_THRESHOLD = 30
