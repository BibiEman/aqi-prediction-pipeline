"""
historical_weather.py

Fetch historical weather data from Open-Meteo API
for multiple Pakistani cities and save it as a CSV.
"""

import requests
import pandas as pd
from pathlib import Path

from src.historical_data.city_coordinates import CITIES


# =====================================================
# Fetch Historical Weather Data for One City
# =====================================================

def fetch_historical_weather(city, latitude, longitude, start_date, end_date):
    """
    Fetch historical hourly weather data from Open-Meteo.

    Parameters:
        city (str): City name
        latitude (float): Latitude
        longitude (float): Longitude
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD

    Returns:
        pandas.DataFrame
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ],
        "timezone": "Asia/Karachi"
    }

    print(f"Fetching weather for {city}...")

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    weather_df = pd.DataFrame({
        "timestamp": hourly["time"],
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "cloud_cover": hourly["cloud_cover"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "precipitation": hourly["precipitation"]
    })

    return weather_df


# =====================================================
# Collect Weather for All Cities
# =====================================================

def collect_all_weather(start_date, end_date):
    """
    Collect historical weather data for all cities.
    """

    all_weather = []

    for city, (latitude, longitude) in CITIES.items():

        weather_df = fetch_historical_weather(
            city,
            latitude,
            longitude,
            start_date,
            end_date
        )

        all_weather.append(weather_df)

    final_df = pd.concat(all_weather, ignore_index=True)

    return final_df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):
    """
    Save dataframe as CSV.
    """

    project_root = Path(__file__).resolve().parents[2]

    output_folder = (
        project_root
        / "data"
        / "historical"
        / "weather"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "historical_weather.csv"

    df.to_csv(output_file, index=False)

    print("\nDataset saved successfully!")

    print(output_file)


# =====================================================
# Main Function
# =====================================================

def main():

    # Approximately 6 months of data
    start_date = "2025-01-01"
    end_date = "2025-06-30"

    print("=" * 60)
    print("Collecting Historical Weather Data")
    print("=" * 60)

    weather_df = collect_all_weather(
        start_date,
        end_date
    )

    print("\nHistorical Weather Dataset\n")

    print(weather_df.head())

    print("\nDataset Shape:", weather_df.shape)

    save_dataset(weather_df)


if __name__ == "__main__":
    main()