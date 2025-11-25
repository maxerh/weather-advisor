import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ProphetEvaluator:
    """
    Evaluator for Prophet forecasts.
    Takes the trained ProphetTrainer (or model + history),
    makes predictions, and computes error metrics.
    """

    def __init__(self, trainer, ds_val: pd.Series, y_val: pd.Series):
        """
        Args:
          trainer: instance of ProphetTrainer (or any object with .prophet inside)
          ds_val: pandas Series of datetimes (validation)
          y_val: pandas Series of true target values (validation)
        """
        self.trainer = trainer
        self.ds_val = pd.to_datetime(ds_val)
        self.y_val = y_val.to_numpy()

    def evaluate(self):
        # Predict
        forecast_df = self.trainer.prophet.predict_df(pd.DataFrame({"ds": self.ds_val}))

        # Align predictions
        merged = pd.merge(
            pd.DataFrame({"ds": self.ds_val, "y_true": self.y_val}),
            forecast_df[["ds", "yhat"]],
            on="ds", how="left"
        )

        # Compute metrics
        mse = mean_squared_error(merged["y_true"], merged["yhat"])
        mae = mean_absolute_error(merged["y_true"], merged["yhat"])
        rmse = np.sqrt(mse)

        # Log metrics to MLflow
        mlflow.log_metric("prophet_mse", mse)
        mlflow.log_metric("prophet_mae", mae)
        mlflow.log_metric("prophet_rmse", rmse)

        print(f"Prophet Evaluation → MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {"mse": mse, "mae": mae, "rmse": rmse}