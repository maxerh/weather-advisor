import torch
from src.evaluators.base_eval import BaseEvaluator
from src.models.lstm_model import LSTMForecastModel


class LSTMEvaluator(BaseEvaluator):
    """
    Evaluator for LSTMForecastModel.
    """
    def __init__(self, model_name, dataloader, device=None):
        super().__init__(
            model_name=model_name,
            dataloader=dataloader,
            device=device,
        )

    def build_model(self, input_dim, output_dim, pred_len, hidden_dim, num_layers):
        return LSTMForecastModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            prediction_length=pred_len,
            output_dim=output_dim,
        )

    def _load_checkpoint(self):
        """Loads model checkpoint (.pt) and rebuilds the model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Rebuild the model using saved parameters
        model_args = {
            "input_dim": checkpoint["input_dim"],
            "output_dim": checkpoint["output_dim"],
            "pred_len": checkpoint["pred_len"],
            "hidden_dim": checkpoint["hidden_dim"],
            "num_layers": checkpoint["num_layers"],
        }
        #self.ckpt_kwargs = model_args

        model = self.build_model(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loaded model from {self.checkpoint_path}")

        return model, model_args

    def predict_step(self, inputs):
        """
        Forward pass.
        """
        return self.model(inputs)

    def compute_metrics(self, y_true, y_pred):
        """
        Compute MAE and RMSE.
        """
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        mse = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }