import mlflow
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.prophet_model import ProphetWrapper


class ProphetTrainer:
    """
    Trainer class for Prophet (not PyTorch). Does fit, predict, evaluate, log metrics.
    """

    def __init__(self, ds: pd.Series, y: pd.Series, freq: str = "H", prophet_kwargs=None):
        """
        ds: pandas Series of datetimes for training
        y: pandas Series of target values (must align with ds)
        freq: frequency for prediction (used in make_future_dataframe)
        prophet_kwargs: dict of Prophet constructor arguments
        """
        self.ds = ds
        self.y = y
        self.freq = freq
        self.prophet = ProphetWrapper(**(prophet_kwargs or {}))
        self.history = None
        self.forecast = None
        mlflow.set_experiment("weather_forecasting")

    def train(self):
        df_train = pd.DataFrame({"ds": self.ds, "y": self.y})
        self.prophet.fit(df_train)
        self.history = df_train
        with open("trained_models/prophet.pkl", "wb") as f:
            pickle.dump(self.prophet, f)

    def predict(self, periods: int):
        self.forecast = self.prophet.predict(periods, freq=self.freq)
        return self.forecast

    def evaluate(self, val_ds: pd.Series, val_y: pd.Series):
        """
        Evaluate Prophet on validation (or test) period.
        val_ds: datetime Series (for validation period)
        val_y: actual values for that period
        Returns:
          dict of metrics
        """
        # Forecast for validation range
        df_pred = pd.DataFrame({"ds": val_ds})
        df_pred["ds"] = pd.to_datetime(df_pred["ds"])
        forecast_df = self.prophet.predict_df(df_pred)

        # Align predictions: merge on ds
        merged = pd.merge(
            pd.DataFrame({"ds": val_ds, "y_true": val_y}),
            forecast_df[["ds", "yhat"]],
            on="ds",
            how="left"
        )

        # Compute metrics
        mse = mean_squared_error(merged["y_true"], merged["yhat"])
        mae = mean_absolute_error(merged["y_true"], merged["yhat"])
        rmse = float(np.sqrt(mse))

        # Log with MLflow
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        return {"mse": mse, "rmse": rmse, "mae": mae}