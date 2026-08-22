"""
model_registry.py

Model registry for the AQI prediction pipeline.

Features:
    - Model versioning
    - Production model tracking
    - Automatic version numbering
    - Model artifact copying
    - Candidate models
    - Automatic model promotion
    - Rollback support
    - Registry stored in JSON
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.model_training.config import (
    MODEL_DIR,
    REGISTRY_FILE,
)


# =====================================================
# Constants
# =====================================================

REGISTRY_DIR = MODEL_DIR / "registry"


# =====================================================
# Registry Helpers
# =====================================================

def load_registry():
    """
    Load registry.json.

    If the registry does not exist, return an empty registry.
    """

    if not REGISTRY_FILE.exists():

        return {
            "models": {}
        }

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            registry = json.load(file)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"Invalid registry file:\n{REGISTRY_FILE}"
        ) from error

    if "models" not in registry:

        registry["models"] = {}

    return registry


# =====================================================
# Save Registry
# =====================================================

def save_registry(registry):
    """
    Save registry to registry.json.
    """

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REGISTRY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


# =====================================================
# Version Handling
# =====================================================

def get_next_version(
    model_name,
    registry=None
):
    """
    Generate the next version.

    Examples:

        No versions -> v1
        v1          -> v2
        v1,v2       -> v3
    """

    if registry is None:
        registry = load_registry()

    model_info = registry.get(
        "models",
        {}
    ).get(
        model_name
    )

    if not model_info:

        return "v1"

    versions = model_info.get(
        "versions",
        []
    )

    if not versions:

        return "v1"

    version_numbers = []

    for item in versions:

        version_string = str(
            item.get(
                "version",
                ""
            )
        ).strip()

        # Accept:
        # v1
        # V1
        # 1

        if version_string.lower().startswith("v"):

            version_string = version_string[1:]

        try:

            number = int(
                version_string
            )

            version_numbers.append(
                number
            )

        except (
            ValueError,
            TypeError
        ):

            continue

    if not version_numbers:

        return "v1"

    next_number = (
        max(version_numbers) + 1
    )

    return f"v{next_number}"


# =====================================================
# Register Model
# =====================================================

def register_model(
    model_name,
    model_path,
    metrics,
    dataset_version="unknown",
):
    """
    Register a trained model.

    Parameters
    ----------
    model_name : str
        Name of model.

    model_path : str or Path
        Path to trained model.

    metrics : dict
        Evaluation metrics.

    dataset_version : str
        Dataset version/date range.

    Returns
    -------
    dict
        Registered model metadata.
    """

    print("\n" + "=" * 70)
    print("REGISTERING MODEL")
    print("=" * 70)

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    registry = load_registry()

    # -------------------------------------------------
    # Create Model Entry
    # -------------------------------------------------

    if model_name not in registry["models"]:

        registry["models"][model_name] = {

            "production_version": None,

            "versions": []
        }

    # -------------------------------------------------
    # Generate Version
    # -------------------------------------------------

    version = get_next_version(
        model_name,
        registry
    )

    print(
        f"Model       : {model_name}"
    )

    print(
        f"Version     : {version}"
    )

    # -------------------------------------------------
    # Create Version Directory
    # -------------------------------------------------

    version_dir = (
        REGISTRY_DIR
        / model_name
        / version
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        version_dir
        / "model.pkl"
    )

    # -------------------------------------------------
    # Copy Model Artifact
    # -------------------------------------------------

    shutil.copy2(
        model_path,
        destination
    )

    # -------------------------------------------------
    # Clean Metrics
    # -------------------------------------------------

    clean_metrics = {

        "MAE": float(
            metrics.get(
                "MAE",
                metrics.get(
                    "mae",
                    0
                )
            )
        ),

        "RMSE": float(
            metrics.get(
                "RMSE",
                metrics.get(
                    "rmse",
                    0
                )
            )
        ),

        "R2": float(
            metrics.get(
                "R2",
                metrics.get(
                    "r2",
                    0
                )
            )
        ),

        "MAPE": float(
            metrics.get(
                "MAPE",
                metrics.get(
                    "MAPE (%)",
                    metrics.get(
                        "mape",
                        0
                    )
                )
            )
        )
    }

    # -------------------------------------------------
    # Timestamp
    # -------------------------------------------------

    registered_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # -------------------------------------------------
    # Version Entry
    # -------------------------------------------------

    version_entry = {

        "model_name": model_name,

        "version": version,

        "registered_at": registered_at,

        "dataset_version": dataset_version,

        "metrics": clean_metrics,

        "model_path": str(
            destination
        ),

        "status": "candidate"
    }

    registry["models"][model_name][
        "versions"
    ].append(
        version_entry
    )

    save_registry(
        registry
    )

    print(
        f"RMSE        : "
        f"{clean_metrics['RMSE']}"
    )

    print(
        f"MAE         : "
        f"{clean_metrics['MAE']}"
    )

    print(
        f"R2          : "
        f"{clean_metrics['R2']}"
    )

    print(
        f"MAPE        : "
        f"{clean_metrics['MAPE']}%"
    )

    print(
        f"Model Path  : "
        f"{destination}"
    )

    print(
        "\nModel registered successfully."
    )

    return version_entry


# =====================================================
# Get Production Model
# =====================================================

def get_production_model(
    model_name
):
    """
    Return production model metadata.
    """

    registry = load_registry()

    model_info = registry.get(
        "models",
        {}
    ).get(
        model_name
    )

    if not model_info:

        return None

    production_version = (
        model_info.get(
            "production_version"
        )
    )

    if not production_version:

        return None

    for version in model_info.get(
        "versions",
        []
    ):

        if (
            version.get("version")
            == production_version
        ):

            return version

    return None


# =====================================================
# Promote Model
# =====================================================

def promote_model(
    model_name,
    version
):
    """
    Promote a model version to production.

    Previous production version is archived.
    """

    registry = load_registry()

    if model_name not in registry["models"]:

        raise ValueError(
            f"Model not found in registry: "
            f"{model_name}"
        )

    versions = registry["models"][
        model_name
    ].get(
        "versions",
        []
    )

    target = None

    # -------------------------------------------------
    # Find Target
    # -------------------------------------------------

    for item in versions:

        if item.get(
            "version"
        ) == version:

            target = item

    if target is None:

        raise ValueError(
            f"Version not found: "
            f"{model_name} {version}"
        )

    # -------------------------------------------------
    # Archive Current Production
    # -------------------------------------------------

    for item in versions:

        if (
            item.get("status")
            == "production"
            and item.get("version")
            != version
        ):

            item["status"] = "archived"

    # -------------------------------------------------
    # Promote Target
    # -------------------------------------------------

    target["status"] = "production"

    registry["models"][
        model_name
    ][
        "production_version"
    ] = version

    save_registry(
        registry
    )

    print("\n" + "=" * 70)
    print("MODEL PROMOTED TO PRODUCTION")
    print("=" * 70)

    print(
        f"Model   : {model_name}"
    )

    print(
        f"Version : {version}"
    )

    return target


# =====================================================
# Register + Automatically Promote
# =====================================================

def register_and_promote_if_better(
    model_name,
    model_path,
    metrics,
    dataset_version="unknown",
):
    """
    Register a new model.

    If no production model exists:
        promote automatically.

    If production exists:
        compare RMSE.

    New model is promoted only when:
        new RMSE < production RMSE
    """

    registered = register_model(
        model_name=model_name,
        model_path=model_path,
        metrics=metrics,
        dataset_version=dataset_version,
    )

    production = get_production_model(
        model_name
    )

    # -------------------------------------------------
    # No Production Model
    # -------------------------------------------------

    if production is None:

        print(
            "\nNo production model exists."
        )

        print(
            f"Promoting "
            f"{model_name} "
            f"{registered['version']} "
            f"to production."
        )

        return promote_model(
            model_name,
            registered["version"]
        )

    # -------------------------------------------------
    # Compare RMSE
    # -------------------------------------------------

    production_rmse = float(
        production["metrics"]["RMSE"]
    )

    candidate_rmse = float(
        registered["metrics"]["RMSE"]
    )

    print("\n" + "=" * 70)
    print("PRODUCTION MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Production : "
        f"{production['version']}"
    )

    print(
        f"Production RMSE : "
        f"{production_rmse:.4f}"
    )

    print(
        f"Candidate  : "
        f"{registered['version']}"
    )

    print(
        f"Candidate RMSE : "
        f"{candidate_rmse:.4f}"
    )

    # -------------------------------------------------
    # Candidate Better
    # -------------------------------------------------

    if candidate_rmse < production_rmse:

        print(
            "\nCandidate model is BETTER."
        )

        print(
            "Promoting candidate to production..."
        )

        return promote_model(
            model_name,
            registered["version"]
        )

    # -------------------------------------------------
    # Candidate Worse
    # -------------------------------------------------

    print(
        "\nCandidate model is NOT better."
    )

    print(
        "Keeping current production model."
    )

    return registered


# =====================================================
# Rollback
# =====================================================

def rollback_model(
    model_name,
    version
):
    """
    Roll back production to a previous version.
    """

    print("\n" + "=" * 70)
    print("MODEL ROLLBACK")
    print("=" * 70)

    production = get_production_model(
        model_name
    )

    if production:

        print(
            f"Current Production : "
            f"{production['version']}"
        )

    print(
        f"Rollback Target    : "
        f"{version}"
    )

    return promote_model(
        model_name,
        version
    )


# =====================================================
# Display Registry
# =====================================================

def display_registry():
    """
    Display all registered models.
    """

    registry = load_registry()

    print("\n" + "=" * 70)
    print("MODEL REGISTRY")
    print("=" * 70)

    if not registry.get(
        "models"
    ):

        print(
            "\nRegistry is empty."
        )

        return

    for model_name, model_info in (
        registry["models"].items()
    ):

        print(
            f"\nModel: {model_name}"
        )

        print(
            f"Production Version: "
            f"{model_info.get('production_version')}"
        )

        print(
            "-" * 60
        )

        for version in model_info.get(
            "versions",
            []
        ):

            metrics = version.get(
                "metrics",
                {}
            )

            print(
                f"Version : "
                f"{version.get('version')}"
            )

            print(
                f"Status  : "
                f"{version.get('status')}"
            )

            print(
                f"RMSE    : "
                f"{metrics.get('RMSE')}"
            )

            print(
                f"MAE     : "
                f"{metrics.get('MAE')}"
            )

            print(
                f"R2      : "
                f"{metrics.get('R2')}"
            )

            print(
                f"MAPE    : "
                f"{metrics.get('MAPE')}"
            )

            print(
                f"Path    : "
                f"{version.get('model_path')}"
            )

            print()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    display_registry()