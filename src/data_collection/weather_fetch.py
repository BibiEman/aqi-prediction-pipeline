"""
weather_fetch.py

Contains functions for fetching weather data from the
OpenWeather Current Weather API.
"""

import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city):
    """
    Fetch weather data for a single city.

    Parameters
    ----------
    city : str
        City name.

    Returns
    -------
    dict
        Weather information.
    """

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    try:

        response = requests.get(BASE_URL, params=params)

        response.raise_for_status()

        data = response.json()

        weather = {

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "city": data["name"],

            "latitude": data["coord"]["lat"],
            "longitude": data["coord"]["lon"],

            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],

            "wind_speed": data["wind"]["speed"],

            "cloud_cover": data["clouds"]["all"],

            "visibility": data["visibility"],

            "weather": data["weather"][0]["description"]

        }

        return weather

    except requests.exceptions.RequestException as e:

        print(f"Error fetching weather for {city}")

        print(e)

        return None


def collect_weather_data(cities):
    """
    Collect weather data for multiple cities.

    Parameters
    ----------
    cities : list

    Returns
    -------
    pandas.DataFrame
    """

    weather_records = []

    for city in cities:

        print(f"Fetching weather for {city}...")

        weather = get_weather(city)

        if weather:

            weather_records.append(weather)

    return pd.DataFrame(weather_records)