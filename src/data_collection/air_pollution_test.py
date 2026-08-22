import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Lahore coordinates
lat = 31.5497
lon = 74.3436

url = "http://api.openweathermap.org/data/2.5/air_pollution"

params = {
    "lat": lat,
    "lon": lon,
    "appid": API_KEY
}

response = requests.get(url, params=params)

print(response.json())