from src.trainers.base_trainer import BaseTrainer
import torch.nn as nn
from src.models.lstm_model import LSTMForecastModel


class LSTMTrainer(BaseTrainer):
    def __init__(self, input_dim, output_dim, pred_len, hidden_dim=64, num_layers=2, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super().__init__(model_name="LSTM", **kwargs)

    def build_model(self):
        return LSTMForecastModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            prediction_length=self.pred_len,
            output_dim=self.output_dim,
        )

    def predict_step(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)