import torch.nn as nn
from src.trainers.base_trainer import BaseTrainer
from src.models.transformer_model import TransformerForecastModel


class TransformerTrainer(BaseTrainer):

    def __init__(
        self,
        input_dim,
        output_dim,
        pred_len,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        super().__init__(model_name="Transformer", **kwargs)

    def build_model(self):
        return TransformerForecastModel(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            prediction_length=self.pred_len,
            output_dim=self.output_dim
        )

    def predict_step(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)