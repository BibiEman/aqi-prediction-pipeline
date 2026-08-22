"""
air_quality.py

Fetch current air-pollution data from the
OpenWeather Air Pollution API.

Important
---------
OpenWeather's `main.aqi` value is its own 1-5 air-quality
index. It is NOT the US AQI value used by the forecasting
pipeline.

The pollutant concentrations returned here are later used
by realtime_monitor.py to calculate an approximate US AQI.
"""

import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

BASE_URL = (
    "https://api.openweathermap.org/"
    "data/2.5/air_pollution"
)

REQUEST_TIMEOUT = 15


# ============================================================
# Configuration Validation
# ============================================================

def validate_api_key() -> None:
    """
    Ensure that the OpenWeather API key is configured.
    """

    if not API_KEY:

        raise RuntimeError(
            "\nOPENWEATHER_API_KEY is not configured.\n\n"
            "Add the following to your .env file:\n\n"
            "OPENWEATHER_API_KEY=your_api_key_here"
        )


# ============================================================
# Fetch Air Quality
# ============================================================

def get_air_quality(
    latitude: float,
    longitude: float,
) -> Optional[Dict]:
    """
    Fetch current pollutant concentrations from OpenWeather.

    Parameters
    ----------
    latitude : float
        Geographic latitude.

    longitude : float
        Geographic longitude.

    Returns
    -------
    dict or None
        Current OpenWeather air-quality index and pollutant
        concentrations.

    Notes
    -----
    Returned pollutant concentrations are expressed in µg/m³
    according to the OpenWeather Air Pollution API.

    The returned `aqi` field is OpenWeather's 1-5 index,
    not US AQI.
    """

    validate_api_key()

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    try:

        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            "Latitude and longitude must be numeric."
        ) from error

    if not -90 <= latitude <= 90:

        raise ValueError(
            f"Invalid latitude: {latitude}"
        )

    if not -180 <= longitude <= 180:

        raise ValueError(
            f"Invalid longitude: {longitude}"
        )

    # --------------------------------------------------------
    # Request parameters
    # --------------------------------------------------------

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": API_KEY,
    }

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:

        print(
            "\nAir-quality API request timed out."
        )

        return None

    except requests.exceptions.ConnectionError:

        print(
            "\nCould not connect to the "
            "OpenWeather Air Pollution API."
        )

        return None

    except requests.exceptions.HTTPError as error:

        status_code = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        print(
            "\nOpenWeather Air Pollution API "
            f"returned HTTP {status_code}."
        )

        if status_code == 401:

            print(
                "Check OPENWEATHER_API_KEY."
            )

        return None

    except requests.exceptions.RequestException as error:

        print(
            "\nError fetching air-quality data:"
        )

        print(error)

        return None

    # --------------------------------------------------------
    # Decode response
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        print(
            "\nOpenWeather returned invalid JSON "
            "for air-quality data."
        )

        return None

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    pollution_records = data.get(
        "list",
        []
    )

    if not pollution_records:

        print(
            "\nOpenWeather returned no "
            "air-quality observations."
        )

        return None

    pollution = pollution_records[0]

    main_data = pollution.get(
        "main",
        {}
    )

    components = pollution.get(
        "components",
        {}
    )

    if not components:

        print(
            "\nAir-quality response does not "
            "contain pollutant components."
        )

        return None

    # --------------------------------------------------------
    # Return normalized output
    # --------------------------------------------------------

    return {
        # OpenWeather's own 1-5 AQI index
        "aqi": main_data.get(
            "aqi"
        ),

        # Pollutants
        "co": components.get(
            "co"
        ),

        "no": components.get(
            "no"
        ),

        "no2": components.get(
            "no2"
        ),

        "o3": components.get(
            "o3"
        ),

        "so2": components.get(
            "so2"
        ),

        "pm2_5": components.get(
            "pm2_5"
        ),

        "pm10": components.get(
            "pm10"
        ),

        "nh3": components.get(
            "nh3"
        ),

        # API observation timestamp if available
        "api_timestamp": pollution.get(
            "dt"
        ),
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("OPENWEATHER AIR QUALITY TEST")
    print("=" * 70)

    # Lahore coordinates
    test_latitude = 31.5497
    test_longitude = 74.3436

    result = get_air_quality(
        latitude=test_latitude,
        longitude=test_longitude,
    )

    if result:

        print(
            "\nAir-quality request successful.\n"
        )

        for key, value in result.items():

            print(
                f"{key:16s}: {value}"
            )

    else:

        print(
            "\nAir-quality request failed."
        )