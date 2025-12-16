import requests
import pandas as pd
from datetime import datetime, timedelta
from datetime import timezone as tz
from pathlib import Path

def fetch_live_weather(
    latitude: float,
    longitude: float,
    hourly_vars: list,
    timezone: str = "auto",
    lookback_hours: int = 200
):
    """
    Fetch recent weather data for live inference.
    Returns a DataFrame compatible with FeatureGenerator.
    """

    base_url = "https://api.open-meteo.com/v1/forecast"

    now = datetime.now(tz.utc)
    start = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
    end   = now.strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "start_date": start,
        "end_date": end,
        "timezone": timezone,
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df
