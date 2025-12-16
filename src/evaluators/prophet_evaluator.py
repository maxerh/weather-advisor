import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.evaluators.base_eval import BaseEvaluator

class ProphetEvaluator:
    def __init__(self, model_name: str, ds_val: pd.Series, y_val: pd.Series):
        self.model_name = model_name
        self.ds_val = pd.to_datetime(ds_val)
        self.y_val = y_val.to_numpy()
        self.checkpoint_path = "trained_models/prophet.pkl"
        self.model, _ = self._load_checkpoint()

    def _load_checkpoint(self):
        with open(self.checkpoint_path, 'rb') as f:
            prophet_model = pickle.load(f)
        return prophet_model, None


    def evaluate(self, visualize: bool = True):
        # TODO: visualization code needed
        forecast_df = self.model.predict(len(self.ds_val))

        merged = pd.merge(
            pd.DataFrame({"ds": self.ds_val, "y_true": self.y_val}),
            forecast_df[["ds", "yhat"]],
            on="ds",
            how="left"
        )

        mse = mean_squared_error(merged["y_true"], merged["yhat"])
        mae = mean_absolute_error(merged["y_true"], merged["yhat"])
        rmse = float(np.sqrt(mse))

        print(f"Prophet evaluation → MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {"mse": mse, "mae": mae, "rmse": rmse}