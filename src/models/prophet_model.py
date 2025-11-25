import pandas as pd
from prophet import Prophet


class ProphetWrapper:
    """
    A simple wrapper around Facebook Prophet to integrate with your forecasting framework.
    This wrapper only deals with univariate (single-target) forecasting.
    """

    def __init__(self, **prophet_kwargs):
        """
        prophet_kwargs: passed to Prophet constructor (e.g. yearly_seasonality, weekly_seasonality)
        """
        self.model = Prophet(**prophet_kwargs)

    def fit(self, df: pd.DataFrame, ds_col: str = "ds", y_col: str = "y"):
        """
        Fit Prophet model.

        Args:
          df: pandas DataFrame with at least two columns: ds_col and y_col
          ds_col: name of datetime column
          y_col: name of target column
        """
        df_prophet = df[[ds_col, y_col]].copy()
        df_prophet = df_prophet.rename(columns={ds_col: "ds", y_col: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        self.model.fit(df_prophet)

    def predict(self, periods: int, freq: str = "H"):
        """
        Make forecast.

        Args:
          periods: number of future steps to forecast
          freq: frequency string (Pandas offset alias, e.g. "H" for hourly, "D" for daily)

        Returns:
          forecast_df: DataFrame including ds, yhat, yhat_lower, yhat_upper
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def predict_df(self, df: pd.DataFrame):
        """
        Forecast for specified dates.

        Args:
          df: pandas DataFrame with ds column (datetime)
        Returns:
          forecast DataFrame with predictions aligned to df['ds']
        """
        df2 = df.copy()
        df2 = df2.rename(columns={df2.columns[0]: "ds"})
        df2["ds"] = pd.to_datetime(df2["ds"])
        forecast = self.model.predict(df2)
        return forecast