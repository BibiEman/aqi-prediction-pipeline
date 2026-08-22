"""
realtime_monitor.py

Real-time AQI and weather monitoring for the
AQI Prediction System.

This module:
    - fetches current weather from OpenWeather
    - fetches current pollutant concentrations
    - calculates US AQI using centralized AQI utilities
    - classifies AQI
    - assigns alert levels
    - creates health guidance
    - supports all project cities
    - normalizes city names
"""

from datetime import datetime
from typing import Dict, List, Optional
import unicodedata

import pandas as pd

from src.data_collection.weather_fetch import (
    get_weather,
)

from src.data_collection.air_quality import (
    get_air_quality,
)

from src.aqi_calculation.calculate_aqi import (
    calculate_us_aqi,
    get_aqi_category,
    get_alert_level,
    get_health_guidance,
)


# ============================================================
# Supported Cities
# ============================================================

SUPPORTED_CITIES = [
    "Faisalabad",
    "Hyderabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rawalpindi",
    "Sialkot",
]


# ============================================================
# Normalize City Name
# ============================================================

def normalize_city_name(
    city: str,
) -> str:
    """
    Normalize city names returned by external APIs.

    Example
    -------
    Siālkot -> Sialkot
    """

    if city is None:
        return "Unknown"

    city = str(
        city
    ).strip()

    normalized = unicodedata.normalize(
        "NFKD",
        city,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )

    lookup = {
        name.lower(): name
        for name in SUPPORTED_CITIES
    }

    return lookup.get(
        normalized.lower(),
        normalized,
    )


# ============================================================
# Validate City
# ============================================================

def validate_city(
    city: str,
) -> str:
    """
    Validate requested city against supported cities.
    """

    normalized_city = normalize_city_name(
        city
    )

    lookup = {
        item.lower(): item
        for item in SUPPORTED_CITIES
    }

    matched_city = lookup.get(
        normalized_city.lower()
    )

    if matched_city is None:

        raise ValueError(
            f"Unsupported city: {city}\n"
            f"Supported cities: {SUPPORTED_CITIES}"
        )

    return matched_city


# ============================================================
# Safe Float
# ============================================================

def safe_float(
    value,
) -> Optional[float]:
    """
    Convert API value to float safely.
    """

    if value is None:
        return None

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# Get Current Conditions
# ============================================================

def get_current_conditions(
    city: str,
) -> Dict:
    """
    Fetch real-time weather and pollution data
    for one supported city.

    Parameters
    ----------
    city : str
        City name.

    Returns
    -------
    dict
        Combined current weather, pollutant,
        AQI, alert and health information.
    """

    # --------------------------------------------------------
    # Validate city
    # --------------------------------------------------------

    requested_city = validate_city(
        city
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather = get_weather(
        requested_city
    )

    if not weather:

        raise RuntimeError(
            "Weather data could not be fetched "
            f"for {requested_city}."
        )

    latitude = safe_float(
        weather.get(
            "latitude"
        )
    )

    longitude = safe_float(
        weather.get(
            "longitude"
        )
    )

    if (
        latitude is None
        or longitude is None
    ):

        raise RuntimeError(
            "Weather response does not contain "
            f"valid coordinates for {requested_city}."
        )

    # --------------------------------------------------------
    # Air quality
    # --------------------------------------------------------

    air_quality = get_air_quality(
        latitude,
        longitude,
    )

    if not air_quality:

        raise RuntimeError(
            "Air-quality data could not be fetched "
            f"for {requested_city}."
        )

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------

    pm2_5 = safe_float(
        air_quality.get(
            "pm2_5"
        )
    )

    pm10 = safe_float(
        air_quality.get(
            "pm10"
        )
    )

    carbon_monoxide = safe_float(
        air_quality.get(
            "co"
        )
    )

    nitrogen_monoxide = safe_float(
        air_quality.get(
            "no"
        )
    )

    nitrogen_dioxide = safe_float(
        air_quality.get(
            "no2"
        )
    )

    ozone = safe_float(
        air_quality.get(
            "o3"
        )
    )

    sulphur_dioxide = safe_float(
        air_quality.get(
            "so2"
        )
    )

    ammonia = safe_float(
        air_quality.get(
            "nh3"
        )
    )

    # --------------------------------------------------------
    # US AQI
    # --------------------------------------------------------

    us_aqi = calculate_us_aqi(
        pm2_5=pm2_5,
        pm10=pm10,
    )

    # --------------------------------------------------------
    # Normalize response city
    # --------------------------------------------------------

    api_city = weather.get(
        "city",
        requested_city,
    )

    normalized_city = normalize_city_name(
        api_city
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "status":
            "success",

        "timestamp":
            timestamp,

        "city":
            normalized_city,

        "latitude":
            latitude,

        "longitude":
            longitude,

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        "temperature":
            safe_float(
                weather.get(
                    "temperature"
                )
            ),

        "humidity":
            safe_float(
                weather.get(
                    "humidity"
                )
            ),

        "pressure":
            safe_float(
                weather.get(
                    "pressure"
                )
            ),

        "wind_speed":
            safe_float(
                weather.get(
                    "wind_speed"
                )
            ),

        "wind_direction":
            safe_float(
                weather.get(
                    "wind_direction"
                )
            ),

        "precipitation":
            safe_float(
                weather.get(
                    "precipitation"
                )
            ),

        "cloud_cover":
            safe_float(
                weather.get(
                    "cloud_cover"
                )
            ),

        "visibility":
            safe_float(
                weather.get(
                    "visibility"
                )
            ),

        "weather":
            weather.get(
                "weather"
            ),

        # ----------------------------------------------------
        # Pollutants
        # ----------------------------------------------------

        "pm2_5":
            pm2_5,

        "pm10":
            pm10,

        "carbon_monoxide":
            carbon_monoxide,

        "nitrogen_monoxide":
            nitrogen_monoxide,

        "nitrogen_dioxide":
            nitrogen_dioxide,

        "ozone":
            ozone,

        "sulphur_dioxide":
            sulphur_dioxide,

        "ammonia":
            ammonia,

        # ----------------------------------------------------
        # OpenWeather AQI index
        # ----------------------------------------------------

        "openweather_aqi_index":
            air_quality.get(
                "aqi"
            ),

        # ----------------------------------------------------
        # Calculated US AQI
        # ----------------------------------------------------

        "current_aqi":
            us_aqi,

        "us_aqi":
            us_aqi,

        "aqi_category":
            get_aqi_category(
                us_aqi
            ),

        "alert_level":
            get_alert_level(
                us_aqi
            ),

        "health_guidance":
            get_health_guidance(
                us_aqi
            ),
    }

    return result


# ============================================================
# Monitor Multiple Cities
# ============================================================

def monitor_all_cities(
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch current AQI and weather conditions
    for multiple cities.

    Parameters
    ----------
    cities : list[str], optional
        Cities to monitor.

    Returns
    -------
    pandas.DataFrame
        Current monitoring observations.
    """

    if cities is None:

        cities = (
            SUPPORTED_CITIES
        )

    records = []

    failed_cities = []

    for city in cities:

        print(
            f"Fetching real-time conditions: "
            f"{city}"
        )

        try:

            result = (
                get_current_conditions(
                    city
                )
            )

            records.append(
                result
            )

        except Exception as error:

            failed_cities.append(
                {
                    "city": city,
                    "error": str(error),
                }
            )

            print(
                f"Failed for {city}: "
                f"{error}"
            )

    dataframe = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Sort output
    # --------------------------------------------------------

    if not dataframe.empty:

        if "timestamp" in dataframe.columns:

            dataframe[
                "timestamp"
            ] = pd.to_datetime(
                dataframe[
                    "timestamp"
                ],
                errors="coerce",
            )

        dataframe = (
            dataframe
            .sort_values(
                [
                    "city",
                    "timestamp",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Failure summary
    # --------------------------------------------------------

    if failed_cities:

        print(
            "\nFailed city requests:"
        )

        for failure in failed_cities:

            print(
                f"  - "
                f"{failure['city']}: "
                f"{failure['error']}"
            )

    return dataframe


# ============================================================
# Display Monitoring Summary
# ============================================================

def display_monitoring_summary(
    dataframe: pd.DataFrame,
):
    """
    Display concise real-time monitoring output.
    """

    if dataframe.empty:

        print(
            "\nNo real-time data could be collected."
        )

        return

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL-TIME MONITORING DATA"
    )

    print(
        "=" * 70
    )

    display_columns = [
        column
        for column in [
            "timestamp",
            "city",
            "current_aqi",
            "aqi_category",
            "pm2_5",
            "pm10",
            "temperature",
            "humidity",
            "alert_level",
        ]
        if column in dataframe.columns
    ]

    print(
        "\n"
        + dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MONITORING SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Cities monitored : "
        f"{len(dataframe)}"
    )

    if "current_aqi" in dataframe.columns:

        valid_aqi = pd.to_numeric(
            dataframe[
                "current_aqi"
            ],
            errors="coerce",
        )

        if valid_aqi.notna().any():

            print(
                f"Average AQI      : "
                f"{valid_aqi.mean():.2f}"
            )

            print(
                f"Maximum AQI      : "
                f"{valid_aqi.max():.2f}"
            )

            print(
                f"Minimum AQI      : "
                f"{valid_aqi.min():.2f}"
            )

    if "alert_level" in dataframe.columns:

        warnings = dataframe[
            dataframe[
                "alert_level"
            ].isin(
                [
                    "Warning",
                    "High Alert",
                    "Emergency",
                ]
            )
        ]

        print(
            f"Warning cities   : "
            f"{len(warnings)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "REAL-TIME AQI MONITORING"
    )

    print(
        "=" * 70
    )

    dataframe = (
        monitor_all_cities()
    )

    display_monitoring_summary(
        dataframe
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()