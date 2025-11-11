import numpy as np
import pandas as pd

class FeatureGenerator():
    def __init__(self, path_with_filename: str):
        self.path_with_file_name = path_with_filename
        self.dataframe = None

    def load_data(self):
        self.dataframe = pd.read_csv(self.path_with_file_name, parse_dates=['time'])

    def create_features(self):
        if self.dataframe is None:
            self.load_data()

        #df_hourly['date'] = df_hourly['time'].dt.date
        # Merge daily sunrise/sunset
        #df_daily['date'] = df_daily['time'].dt.date  # if original daily has time as midnight
        #df_hourly = df_hourly.merge(df_daily[['date', 'sunrise', 'sunset', 'daylight_duration']], on='date', how='left')

        # Derive features
        self.dataframe['date'] = self.dataframe['time'].dt.date

        self.dataframe['hour_of_day'] = self.dataframe['time'].dt.hour
        self.dataframe['sunrise'] = pd.to_datetime(self.dataframe['sunrise'])
        self.dataframe['sunset'] = pd.to_datetime(self.dataframe['sunset'])
        self.dataframe['minutes_since_sunrise'] = (self.dataframe['time'] - self.dataframe['sunrise']).dt.total_seconds() / 60
        self.dataframe['minutes_until_sunset'] = (self.dataframe['sunset'] - self.dataframe['time']).dt.total_seconds() / 60
        self.dataframe['is_daylight'] = (
                    (self.dataframe['time'] >= self.dataframe['sunrise']) &
                    (self.dataframe['time'] <= self.dataframe['sunset'])).astype(int)

        # Optionally clip negative values:
        self.dataframe['minutes_since_sunrise'] = self.dataframe['minutes_since_sunrise'].clip(lower=0)
        self.dataframe['minutes_until_sunset'] = self.dataframe['minutes_until_sunset'].clip(lower=0)

        # Add cyclical encoding for hour_of_day:
        self.dataframe['hour_sin'] = np.sin(2 * np.pi * self.dataframe['hour_of_day'] / 24)
        self.dataframe['hour_cos'] = np.cos(2 * np.pi * self.dataframe['hour_of_day'] / 24)

        # Convert wind direction from ° to u/v components
        wd_rad = np.deg2rad(self.dataframe['wind_direction_10m'])
        self.dataframe['wind_u'] = self.dataframe['wind_speed_10m'] * np.cos(wd_rad)
        self.dataframe['wind_v'] = self.dataframe['wind_speed_10m'] * np.sin(wd_rad)
        print(self.dataframe.head())


if __name__ == "__main__":
    feature_gen = FeatureGenerator("../../data/history/munich_weather_2015_2024.csv")
    feature_gen.create_features()
