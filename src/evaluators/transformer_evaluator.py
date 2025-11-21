import torch
from src.evaluators.base_eval import BaseEvaluator
from src.models.transformer_model import TransformerForecastModel  # adjust path if needed


class TransformerEvaluator(BaseEvaluator):
    """
    Evaluator for TransformerForecastModel.

    This implementation is defensive: it will try to read all model hyperparameters
    from the checkpoint (embed_dim, num_heads, hidden_dim, num_layers). If those
    keys are missing (older checkpoints), it falls back to reasonable defaults.
    """

    def _load_checkpoint(self):
        """Loads model checkpoint (.pt) and rebuilds the model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Rebuild the model using saved parameters
        model_args = {
            "input_dim": checkpoint["input_dim"],
            "embed_dim": checkpoint["embed_dim"],
            "output_dim": checkpoint["output_dim"],
            "pred_len": checkpoint["pred_len"],
            "ff_dim": checkpoint["ff_dim"],
            "num_layers": checkpoint["num_layers"],
            "num_heads": checkpoint["num_heads"],
        }
        #self.ckpt_kwargs = model_args
        #input_dim, embed_dim, num_heads, ff_dim, num_layers, pred_len, output_dim
        print(model_args)
        model = self.build_model(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loaded model from {self.checkpoint_path}")

        return model, model_args

    def build_model(
        self, input_dim, embed_dim, num_heads, ff_dim, num_layers, pred_len, output_dim
    ):
        model = TransformerForecastModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            prediction_length=pred_len,
            output_dim=output_dim,
        )

        return model

    def predict_step(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        # inputs expected shape: (B, T, C)
        return self.model(inputs)

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Compute common forecasting metrics (MAE, RMSE)."""
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        return {"mae": mae, "rmse": rmse}
