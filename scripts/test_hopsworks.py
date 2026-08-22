import os
import hopsworks
from pathlib import Path

PROJECT_NAME = "AQI_Prediction_3days"


def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable is not set."
        )

    cert_folder = Path.home() / ".hopsworks" / "certs"
    cert_folder.mkdir(parents=True, exist_ok=True)

    print("Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=api_key,
        cert_folder=str(cert_folder),
        engine="python",
    )

    print()
    print("Hopsworks connection successful!")
    print(f"Project: {project.name}")


if __name__ == "__main__":
    main()