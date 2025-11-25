import torch
import torch.nn as nn
from src.trainers.base_trainer import BaseTrainer
from src.models.autoformer_model import Autoformer


class AutoformerTrainer(BaseTrainer):
    def __init__(
        self,
        input_dim,
        output_dim,
        input_length,
        pred_len,
        d_model=128,
        n_heads=8,
        ff_dim=256,
        num_layers=3,
        kernel_size=25,
        top_k=5,
        dropout=0.1,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.kernel_size = kernel_size
        self.top_k = top_k
        self.dropout = dropout

        super().__init__(model_name="autoformer", **kwargs)

    def build_model(self):
        return Autoformer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length,
            pred_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            kernel_size=self.kernel_size,
            top_k=self.top_k,
            dropout=self.dropout,
        )

    def predict_step(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)