import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.evaluators.base_eval import BaseEvaluator

class ProphetEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, ds_val: pd.Series, y_val: pd.Series):
        self.model_name = model_name
        self.ds_val = pd.to_datetime(ds_val)
        self.y_val = y_val.to_numpy()

    def _load_checkpoint(self):
        with open(self.checkpoint_path, 'rb') as f:
            self.prophet_model = pickle.load(f)
        return self.prophet_model, None


    def evaluate(self, visualize: bool = True):
        # TODO: visualization code needed
        future_df = pd.DataFrame({"ds": self.ds_val})
        forecast_df = self.prophet_model.predict(future_df)

        merged = pd.merge(
            pd.DataFrame({"ds": self.ds_val, "y_true": self.y_val}),
            forecast_df[["ds", "yhat"]],
            on="ds",
            how="left"
        )

        mse = mean_squared_error(merged["y_true"], merged["yhat"])
        mae = mean_absolute_error(merged["y_true"], merged["yhat"])
        rmse = np.sqrt(mse)

        mlflow.log_metric("prophet_mse", mse)
        mlflow.log_metric("prophet_mae", mae)
        mlflow.log_metric("prophet_rmse", rmse)

        print(f"Prophet evaluation → MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {"mse": mse, "mae": mae, "rmse": rmse}