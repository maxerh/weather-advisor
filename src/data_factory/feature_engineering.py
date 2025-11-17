import numpy as np
import pandas as pd

from src.visualizations.visualizer import Visualizer

class FeatureGenerator(Visualizer):
    def __init__(self, path_with_filename: str):
        super().__init__()
        self.path_with_file_name = path_with_filename
        self.dataframe = None
        self.stats = None
        self.numeric_cols = None
        self.non_numeric = None


    def load_data(self):
        """
        Load weather data from CSV file into a pandas DataFrame.
        """
        self.dataframe = pd.read_csv(self.path_with_file_name, parse_dates=['time'])
        self.numeric_cols = self.dataframe.select_dtypes(include=np.number).columns
        self.non_numeric = self.dataframe.drop(columns=self.numeric_cols)

        if self.stats is None:
            self.stats = self.get_column_stats(self.dataframe.select_dtypes(include=[np.number]))

    def create_features(self):
        """
        Create time-based and weather-related features from the loaded dataframe.
        """
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
        #print(self.dataframe.head())

        self.dataframe.drop(columns=['sunrise', 'sunset', 'date'], inplace=True)
        self.numeric_cols = self.dataframe.select_dtypes(include=np.number).columns
        self.non_numeric = self.dataframe.drop(columns=self.numeric_cols)
        self.stats = self.get_column_stats(self.dataframe.select_dtypes(include=[np.number]))

    def get_column_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic statistics for each column in the DataFrame."""
        stats = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'median': df.median()
        })
        return stats


    def rolling_mean(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Compute rolling mean for numeric columns in the DataFrame."""
        # Compute rolling mean on those columns
        rolled = df[self.numeric_cols].rolling(window=window, min_periods=1).mean()
        return pd.concat([self.non_numeric, rolled], axis=1)

    def rolling_mean_time_based(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        """Compute rolling mean for numeric columns in the DataFrame based on time window."""
        # Make sure the index is a datetime index
        df = df.set_index('time')  # if time is a column
        # Only roll numeric columns
        rolled = df[self.numeric_cols].rolling(window=window, min_periods=1).mean()
        return rolled.reset_index()

    def z_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-normalization (standardization) to numeric columns only.
        Non-numeric columns are preserved.
        """
        # It assumes self.stats['mean'] and self.stats['std'] are aligned with numeric_cols
        df_numeric = (df[self.numeric_cols] - self.stats['mean'][self.numeric_cols]) / self.stats['std'][self.numeric_cols]
        result = pd.concat([self.non_numeric, df_numeric], axis=1)
        return result[df.columns]

    def normalize_min_max(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Min-Max normalization to the DataFrame."""
        df_numeric = (df[self.numeric_cols] - self.stats['min'][self.numeric_cols]) / (self.stats['max'][self.numeric_cols] - self.stats['min'][self.numeric_cols])
        result = pd.concat([self.non_numeric, df_numeric], axis=1)
        return result[df.columns]

    def normalize_negone_one(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame values to the range [-1, 1]."""
        df_numeric = 2 * (df[self.numeric_cols] - self.stats['min'][self.numeric_cols]) / (self.stats['max'][self.numeric_cols]- self.stats['min'][self.numeric_cols]) - 1
        result = pd.concat([self.non_numeric, df_numeric], axis=1)
        return result[df.columns]

    def normalize_zero_one(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame values to the range [0, 1]. Same as min-max normalization."""
        return self.normalize_min_max(df)


if __name__ == "__main__":
    feature_gen = FeatureGenerator("../../data/history/munich_weather_2015_2024.csv")
    feature_gen.create_features()
    #feature_gen.visualize_single_channel(feature_gen.dataframe[['temperature_2m']].values,
    #                                     channel=0, channel_first=False,
    #                                     title="Temperature Over Time", xlabel="Time", ylabel="Temperature (°C)")

    channels = feature_gen.dataframe.columns.tolist()
    print(channels)
    #c=1
    #feature_gen.visualize_single_channel(feature_gen.dataframe,
    #                                     channel=c, channel_first=False,
    #                                     title=f"{channels[c]}Over Time", xlabel="Time", ylabel="")

    feature_gen.visualize_multiple_channels(feature_gen.normalize_min_max(feature_gen.dataframe),
                                            channels=[1,7,21], channel_first=False,
                                            title="", xlabel="Time", ylabel="")