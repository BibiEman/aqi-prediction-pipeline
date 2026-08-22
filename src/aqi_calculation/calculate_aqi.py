"""
calculate_aqi.py

US AQI calculation utilities.

Currently calculates AQI using:
    - PM2.5
    - PM10

The final AQI is the maximum pollutant sub-index.
"""

from typing import Optional


# ============================================================
# AQI Breakpoints
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


# ============================================================
# Calculate Pollutant Sub-Index
# ============================================================

def calculate_sub_index(
    concentration: float,
    breakpoints,
) -> Optional[int]:
    """
    Convert pollutant concentration into AQI sub-index.
    """

    if concentration is None:
        return None

    try:
        concentration = float(concentration)

    except (TypeError, ValueError):
        return None

    if concentration < 0:
        return None

    for (
        c_low,
        c_high,
        i_low,
        i_high,
    ) in breakpoints:

        if c_low <= concentration <= c_high:

            aqi = (
                (i_high - i_low)
                / (c_high - c_low)
                * (concentration - c_low)
                + i_low
            )

            return int(
                round(aqi)
            )

    if concentration > breakpoints[-1][1]:
        return 500

    return None


# ============================================================
# Calculate US AQI
# ============================================================

def calculate_us_aqi(
    pm2_5: float,
    pm10: float,
) -> Optional[int]:
    """
    Calculate US AQI using PM2.5 and PM10.

    Final AQI = maximum available pollutant sub-index.
    """

    pm25_aqi = calculate_sub_index(
        pm2_5,
        PM25_BREAKPOINTS,
    )

    pm10_aqi = calculate_sub_index(
        pm10,
        PM10_BREAKPOINTS,
    )

    valid_values = [
        value
        for value in [
            pm25_aqi,
            pm10_aqi,
        ]
        if value is not None
    ]

    if not valid_values:
        return None

    return max(
        valid_values
    )


# ============================================================
# AQI Category
# ============================================================

def get_aqi_category(
    aqi: Optional[float],
) -> str:

    if aqi is None:
        return "Unknown"

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# Alert Level
# ============================================================

def get_alert_level(
    aqi: Optional[float],
) -> str:

    if aqi is None:
        return "Unknown"

    if aqi <= 100:
        return "Normal"

    if aqi <= 150:
        return "Caution"

    if aqi <= 200:
        return "Warning"

    if aqi <= 300:
        return "High Alert"

    return "Emergency"


# ============================================================
# Health Guidance
# ============================================================

def get_health_guidance(
    aqi: Optional[float],
) -> str:

    if aqi is None:

        return (
            "Current AQI is unavailable."
        )

    if aqi <= 50:

        return (
            "Air quality is satisfactory. "
            "Normal outdoor activities may continue."
        )

    if aqi <= 100:

        return (
            "Air quality is acceptable for most people. "
            "Very sensitive individuals should monitor exposure."
        )

    if aqi <= 150:

        return (
            "Sensitive groups should reduce prolonged "
            "or strenuous outdoor activity."
        )

    if aqi <= 200:

        return (
            "Everyone may begin to experience health effects. "
            "Sensitive groups should avoid prolonged outdoor exposure."
        )

    if aqi <= 300:

        return (
            "Health alert conditions are expected. "
            "Significantly reduce outdoor activity."
        )

    return (
        "Hazardous conditions. Avoid unnecessary outdoor "
        "activity and follow public-health guidance."
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AQI CALCULATION TEST")
    print("=" * 70)

    pm25 = 36.76
    pm10 = 111.54

    aqi = calculate_us_aqi(
        pm2_5=pm25,
        pm10=pm10,
    )

    print(
        f"\nPM2.5 : {pm25}"
    )

    print(
        f"PM10  : {pm10}"
    )

    print(
        f"AQI   : {aqi}"
    )

    print(
        f"Category: "
        f"{get_aqi_category(aqi)}"
    )