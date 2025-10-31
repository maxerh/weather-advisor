import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path


current_file = Path(__file__).resolve()
src_folder   = current_file.parent      # = .../project/src/data_factory
project_root = src_folder.parent.parent # = .../project

def download_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: list,
    daily_vars: list = None,
    timezone: str = "auto",
    output_csv: str = "weather_history.csv"
):
    """
    Download historical weather data from Open-Meteo and save to CSV.
    Args:
      latitude, longitude: location coordinates
      start_date, end_date: ISO date strings e.g. "2015-01-01"
      hourly_vars: list of hourly variables (e.g. ["temperature_2m","relative_humidity_2m"])
      daily_vars: optional list of daily variables
      timezone: timezone string (e.g., "Europe/Berlin") or "auto"
      output_csv: output filename
    """

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    if hourly_vars:
        params["hourly"] = ",".join(hourly_vars)
    if daily_vars:
        params["daily"] = ",".join(daily_vars)

    print("Requesting URL with params:", params)
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    # Extract time-series into DataFrame
    df_hourly = pd.DataFrame(data["hourly"])
    df_daily = pd.DataFrame(data["daily"])

    # Combine if needed (example: hourly + daily join)

    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_hourly['date'] = df_hourly['time'].dt.date
    df_daily['date'] = df_daily['time'].dt.date
    df_merged = df_hourly.merge(df_daily, on='date', how='left', suffixes=('', '_daily'))
    df_merged = df_merged.drop(['date','time_daily'], axis=1)

    # Export to CSV
    data_name = project_root / "data" / "history" / output_csv
    df_merged.to_csv(data_name, index=False)
    print(f"Saved data to {data_name}, rows: {len(df_merged)}")

if __name__ == "__main__":
    # Example usage for Munich: Lat=48.1351, Lon=11.5820
    download_historical_weather(
        latitude=48.1351,
        longitude=11.5820,
        start_date="2015-01-01",
        end_date="2024-12-31",
        hourly_vars=["temperature_2m","relative_humidity_2m","dew_point_2m","wind_speed_10m","wind_direction_10m",
                     "surface_pressure","rain","precipitation", "cloudcover_low", "cloudcover_mid", "cloudcover_high"],
        daily_vars=["temperature_2m_max","temperature_2m_min","precipitation_sum","sunrise", "sunset"],
        timezone="Europe/Berlin",
        output_csv="munich_weather_2015_2024.csv"
    )