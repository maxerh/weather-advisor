import torch.nn as nn

class LSTMForecastModel(nn.Module):
    """
    A simple but strong LSTM forecasting model.

    Inputs:
        x: (batch, input_length, input_dim)
    Outputs:
        y: (batch, prediction_length, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, prediction_length, output_dim, *args, **kwargs):
        super().__init__()

        self.pred_len = prediction_length
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)                # (B, T, H)
        last = out[:, -1, :]                 # (B, H)
        preds = self.fc(last)                # (B, output_dim)

        # Expand prediction to multi-step output:
        preds = preds.unsqueeze(1).repeat(1, self.pred_len, 1)
        return preds