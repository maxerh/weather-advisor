import torch
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint


class BaseEvaluator(ABC):
    """
    Base evaluator class that:
    - loads a trained model checkpoint
    - runs evaluation or forecasting
    - computes metrics
    - supports MLflow logging
    - handles visualization

    Subclasses must implement:
        - build_model()
        - predict_step()
        - compute_metrics()
    """

    def __init__(
        self,
        model_name: str,
        dataloader,
        checkpoint_path: str = "trained_models/",
        mlflow_experiment: str = "weather_forecasting",
        device: str = None,
    ):
        self.model_name = model_name
        self.dataloader = dataloader
        self.checkpoint_path = Path(checkpoint_path+self.model_name+'.pt')

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # MLflow settings
        mlflow.set_experiment(mlflow_experiment)

        # Load checkpoint + instantiate model
        self.model, self.ckpt_kwargs = self._load_checkpoint()

    # ---------------------------------------------------------
    # Abstract methods (mirror BaseTrainer)
    # ---------------------------------------------------------

    @abstractmethod
    def build_model(self, **kwargs):
        """Return model instance with correct architecture."""
        pass

    @abstractmethod
    def predict_step(self, inputs):
        """Forward pass for prediction."""
        pass

    @abstractmethod
    def compute_metrics(self, y_true, y_pred) -> dict:
        """Return a dict of metrics, e.g., {'mae': ..., 'rmse': ...}."""
        pass


    # ---------------------------------------------------------
    # Full Evaluation
    # ---------------------------------------------------------

    def evaluate(self, visualize: bool = True):
        """Evaluate model on entire dataloader & compute metrics."""

        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_name)

            all_preds = []
            all_targets = []
            all_inputs = []

            with torch.no_grad():
                for batch in self.dataloader:
                    inputs, targets = [x.to(self.device) for x in batch]

                    preds = self.predict_step(inputs)

                    all_inputs.append(inputs.cpu())
                    all_targets.append(targets.cpu())
                    all_preds.append(preds.cpu())

            # Concatenate all batches
            all_inputs = torch.cat(all_inputs)
            all_targets = torch.cat(all_targets)
            all_preds = torch.cat(all_preds)

            # Compute metrics
            metrics = self.compute_metrics(all_targets, all_preds)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            print("\nEvaluation Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            if visualize:
                self.visualize_forecast(all_inputs, all_targets, all_preds)

            return metrics

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    def visualize_forecast(self, x, y_true, y_pred, idx: int = 0):
        """
        Plots:
            - historical inputs
            - true future targets
            - predicted future values

        Args:
            idx: sample index to visualize
        """
        x = x[idx, :, 0]             # pick first feature (e.g., temperature)
        future_true = y_true[idx, :, 0]
        future_pred = y_pred[idx, :, 0]

        t_past = range(len(x))
        t_future = range(len(x), len(x) + len(future_true))

        plt.figure(figsize=(12, 5))
        plt.plot(t_past, x, label="History")
        plt.plot(t_future, future_true, label="True Future")
        plt.plot(t_future, future_pred, label="Predicted")
        plt.title(f"Forecast Visualization — {self.model_name}")
        plt.grid()
        plt.legend()
        plt.show()
