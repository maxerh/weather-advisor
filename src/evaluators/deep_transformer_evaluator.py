import torch

from src.models.deep_transformer_model import DeepTransformerForecastModel
from src.evaluators.base_eval import BaseEvaluator

class DeepTransformerEvaluator(BaseEvaluator):

    def _load_checkpoint(self):
        """Loads model checkpoint (.pt) and rebuilds the model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Rebuild the model using saved parameters
        model_args = {
            "input_dim": checkpoint["input_dim"],
            "input_length": checkpoint["input_length"],
            "embed_dim": checkpoint["embed_dim"],
            "output_dim": checkpoint["output_dim"],
            "pred_len": checkpoint["pred_len"],
            "ff_dim": checkpoint["ff_dim"],
            "num_layers": checkpoint["num_layers"],
            "num_heads": checkpoint["num_heads"],
        }
        #self.ckpt_kwargs = model_args
        model = self.build_model(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loaded model from {self.checkpoint_path}")

        return model, model_args

    def build_model(self, **kwargs):
        return DeepTransformerForecastModel(**kwargs)

    def predict_step(self, inputs):
        return self.model(inputs)

    def compute_metrics(self, y_true, y_pred):
        mae = torch.mean(torch.abs(y_true - y_pred))
        mse = torch.mean((y_true - y_pred)**2)
        rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
        return {"mae": mae.item(), "rmse": rmse.item(), "mse": mse.item()}