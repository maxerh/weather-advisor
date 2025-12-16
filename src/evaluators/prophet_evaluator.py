import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
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

        if visualize:
            self.visualize_forecast(merged["y_true"], merged["yhat"])

        return {"mse": mse, "mae": mae, "rmse": rmse}

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    def visualize_forecast(self, y_true, y_pred):
        """
        Plots:
            - historical inputs
            - true future targets
            - predicted future values

        Args:
            idx: sample index to visualize
        """

        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"Forecast Visualization — {self.model_name}")
        plt.grid()
        plt.legend()
        plt.show()
