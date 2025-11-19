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

    def detect_binary_columns(self, df):
        binary_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            unique_values = df[col].dropna().unique()
            if set(unique_values).issubset({0, 1}):
                binary_cols.append(col)
        return binary_cols


    def rolling_mean(self, df: pd.DataFrame, window: int, include_binary: bool = False, binary_mode: str = "proportion") -> pd.DataFrame:
        """
        Compute rolling mean for numeric columns in the DataFrame.

        Args:
            df: input dataframe (must contain columns known to this FeatureGenerator).
            window: integer window size (number of rows) for rolling.
            include_binary: if False (default) binary columns are excluded.
                            if True and binary_mode == "proportion", rolling mean of binary columns
                            will be computed (interpreted as fraction of 1s in the window).
            binary_mode: currently only "proportion" supported when include_binary=True.
        Returns:
            DataFrame with non-numeric columns preserved and rolling aggregates for numeric columns.
        """
        # Ensure metadata exists
        if self.numeric_cols is None:
            # fallback: detect numeric columns on the provided df
            numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols_all = list(self.numeric_cols) + list(self.binary_cols)  # all numeric we know about

        # Determine which numeric columns to roll
        if include_binary:
            cols_to_roll = [c for c in numeric_cols_all if c in df.columns]
        else:
            # only continuous numeric columns (exclude binary)
            continuous_cols = [c for c in (self.numeric_cols or []) if c in df.columns]
            cols_to_roll = continuous_cols

        # If there are no columns to roll, return original df
        if len(cols_to_roll) == 0:
            return df.copy()

        # Perform rolling on selected columns (preserve index & non-numeric columns)
        rolled = df[cols_to_roll].rolling(window=window, min_periods=1).mean()

        # If binary columns were requested but user wants a different mode in the future, handle here
        # (currently proportion is same as mean for binary 0/1)
        # Reassemble: keep non-numeric and columns not rolled (e.g., if binary excluded)
        other_cols = [c for c in df.columns if c not in cols_to_roll]
        result = pd.concat([df[other_cols].reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)

        # Reorder to original columns: put the rolled columns back in their original positions
        # If some rolled columns replaced original ones, we keep result columns in the same order as df.columns
        # For new column values (rolled) we overwrite original numeric columns
        result = result[df.columns.intersection(result.columns).tolist()]  # ensure order & subset

        # For safety, if any columns are missing (shouldn't happen), append them from original df
        for c in df.columns:
            if c not in result.columns:
                result[c] = df[c].values

        return result[df.columns]

    def rolling_mean_time_based(self, df: pd.DataFrame, window: str, include_binary: bool = False, binary_mode: str = "proportion") -> pd.DataFrame:
        """
        Compute rolling mean for numeric columns in the DataFrame using a time-based window.

        Args:
            df: input dataframe (must contain a 'time' column or index)
            window: pandas offset alias or duration string, e.g., '24H', '7D'
            include_binary: if False (default) binary columns are excluded.
            binary_mode: currently only "proportion" supported when include_binary=True.

        Returns:
            DataFrame with an explicit 'time' column reset and rolling aggregates for numeric columns.
        """
        # Ensure 'time' exists
        if 'time' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a 'time' column or a DatetimeIndex for time-based rolling.")

        # Set index to time if not already
        df_time = df.set_index('time') if 'time' in df.columns else df.copy()
        if not isinstance(df_time.index, pd.DatetimeIndex):
            df_time.index = pd.to_datetime(df_time.index)

        # Compose list of columns to roll (same logic as in rolling_mean)
        if self.numeric_cols is None:
            numeric_cols_all = df_time.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols_all = list(self.numeric_cols) + list(self.binary_cols)

        if include_binary:
            cols_to_roll = [c for c in numeric_cols_all if c in df_time.columns]
        else:
            continuous_cols = [c for c in (self.numeric_cols or []) if c in df_time.columns]
            cols_to_roll = continuous_cols

        if len(cols_to_roll) == 0:
            # nothing to roll: return reset index frame
            return df_time.reset_index()

        rolled = df_time[cols_to_roll].rolling(window=window, min_periods=1).mean()

        # Reassemble with non-rolled columns preserved
        other_cols = [c for c in df_time.columns if c not in cols_to_roll]
        result = pd.concat([df_time[other_cols].reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)

        # Try to keep original column order
        result = result[df_time.reset_index().columns.intersection(result.columns).tolist()]

        # Re-add time column if it was index
        if 'time' not in result.columns and isinstance(df_time.index, pd.DatetimeIndex):
            result.insert(0, 'time', df_time.index.to_series().reset_index(drop=True))

        # Fill any missing original columns (safety)
        for c in df.reset_index().columns if 'time' in df.columns else df.columns:
            if c not in result.columns:
                result[c] = df[c].values

        return result[df.columns]

    def train_val_test_split(self, train_ratio=0.7, val_ratio=0.15):
        """
        Perform a chronological train-val-test split.
        The remaining portion is used as test.
        """
        if self.dataframe is None:
            raise ValueError("Data must be loaded or features created before splitting.")

        df = self.dataframe.sort_values("time")

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        return df_train, df_val, df_test

    def fit_normalization(self, df_train):
        """
        Fit normalization parameters (mean, std, min, max) using only the training data.
        Detects binary columns automatically.
        """
        numeric = df_train.select_dtypes(include=[np.number])

        # Detect binary columns
        self.binary_cols = self.detect_binary_columns(df_train)

        # Continuous numeric columns = numeric - binary
        self.numeric_cols = [c for c in numeric.columns if c not in self.binary_cols]

        # Save non-numeric columns (strings/dates, etc.)
        self.non_numeric_cols = df_train.drop(columns=numeric.columns).columns.tolist()

        # Stats only for continuous features
        if len(self.numeric_cols) > 0:
            self.stats = self.get_column_stats(df_train[self.numeric_cols])
        else:
            self.stats = None

    def transform(self, df, method="z"):
        """
        Apply a chosen normalization method. Binary columns remain untouched.
        """
        if self.stats is None:
            raise ValueError("Normalization has not been fitted yet (call fit_normalization first).")

        df_cont = df[self.numeric_cols] if self.numeric_cols else pd.DataFrame(index=df.index)
        df_bin = df[self.binary_cols] if self.binary_cols else pd.DataFrame(index=df.index)
        df_non = df[self.non_numeric_cols] if self.non_numeric_cols else pd.DataFrame(index=df.index)

        if not df_cont.empty:
            if method == "z":
                df_cont_norm = (df_cont - self.stats["mean"]) / self.stats["std"]
            elif method == "minmax":
                df_cont_norm = (df_cont - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
            elif method == "zeroone":
                df_cont_norm = (df_cont - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
            elif method == "negoneone":
                df_cont_norm = 2 * (df_cont - self.stats["min"]) / (self.stats["max"] - self.stats["min"]) - 1
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        else:
            df_cont_norm = pd.DataFrame(index=df.index)

        # Reassemble all columns in original order
        df_out = pd.concat([df_non, df_bin, df_cont_norm], axis=1)
        return df_out[df.columns]


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

    #feature_gen.visualize_multiple_channels(feature_gen.normalize_min_max(feature_gen.dataframe),
    #                                        channels=[1,7,21], channel_first=False,
    #                                        title="", xlabel="Time", ylabel="")
    feature_gen.visualize_multiple_channels(feature_gen.rolling_mean(feature_gen.dataframe, window=100),
                                            channels=[1,7,21], channel_first=False,
                                            title="", xlabel="Time", ylabel="")