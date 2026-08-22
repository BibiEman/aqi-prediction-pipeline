"""
historical_air_quality.py

Fetch historical air quality data from Open-Meteo API
for multiple Pakistani cities and save it as a CSV.
"""

import requests
import pandas as pd
from pathlib import Path

from src.historical_data.city_coordinates import CITIES


# =====================================================
# Fetch Historical Air Quality Data for One City
# =====================================================

def fetch_historical_air_quality(city, latitude, longitude, start_date, end_date):
    """
    Fetch historical hourly air quality data from Open-Meteo.

    Parameters:
        city (str): City name
        latitude (float): Latitude
        longitude (float): Longitude
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD

    Returns:
        pandas.DataFrame
    """

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "timezone": "Asia/Karachi"
    }

    print(f"Fetching air quality for {city}...")

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    air_quality_df = pd.DataFrame({
        "timestamp": hourly["time"],
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "pm10": hourly["pm10"],
        "pm2_5": hourly["pm2_5"],
        "carbon_monoxide": hourly["carbon_monoxide"],
        "nitrogen_dioxide": hourly["nitrogen_dioxide"],
        "sulphur_dioxide": hourly["sulphur_dioxide"],
        "ozone": hourly["ozone"],
        "us_aqi": hourly["us_aqi"]
    })

    return air_quality_df


# =====================================================
# Collect Air Quality for All Cities
# =====================================================

def collect_all_air_quality(start_date, end_date):
    """
    Collect historical air quality data for all cities.
    """

    all_air_quality = []

    for city, (latitude, longitude) in CITIES.items():

        air_quality_df = fetch_historical_air_quality(
            city,
            latitude,
            longitude,
            start_date,
            end_date
        )

        all_air_quality.append(air_quality_df)

    final_df = pd.concat(all_air_quality, ignore_index=True)

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
        / "air_quality"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "historical_air_quality.csv"

    df.to_csv(output_file, index=False)

    print("\nDataset saved successfully!")
    print(output_file)


# =====================================================
# Main Function
# =====================================================

def main():

    start_date = "2025-01-01"
    end_date = "2025-06-30"

    print("=" * 60)
    print("Collecting Historical Air Quality Data")
    print("=" * 60)

    air_quality_df = collect_all_air_quality(
        start_date,
        end_date
    )

    print("\nHistorical Air Quality Dataset\n")
    print(air_quality_df.head())

    print("\nDataset Shape:", air_quality_df.shape)

    save_dataset(air_quality_df)


if __name__ == "__main__":
    main()