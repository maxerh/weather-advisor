import torch.nn as nn
from src.trainers.base_trainer import BaseTrainer
from src.models.deep_transformer_model import DeepTransformerForecastModel


class TransformerForecastTrainer(BaseTrainer):

    def __init__(
        self,
        input_dim,
        output_dim,
        input_length,
        pred_len,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_layers=4,
        dropout=0.1,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout

        super().__init__(model_name="deeptransformer", **kwargs)

    def build_model(self):
        return DeepTransformerForecastModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length,
            pred_len=self.pred_len,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def predict_step(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)