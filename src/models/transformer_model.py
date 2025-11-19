import torch.nn as nn

class TransformerForecastModel(nn.Module):
    """
    Transformer Encoder forecasting model.
    Uses attention over the input window and decoder is a simple MLP head.
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        prediction_length,
        output_dim
    ):
        super().__init__()

        self.pred_len = prediction_length
        self.output_dim = output_dim

        # Input projection
        self.embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Forecasting head
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = self.embedding(x)               # (B, T, embed)
        encoded = self.transformer(x)       # (B, T, embed)

        # Use last timestep representation for prediction
        last = encoded[:, -1, :]            # (B, embed)
        pred = self.fc(last)                # (B, output_dim)

        # Multi-step replication head (simple baseline)
        pred = pred.unsqueeze(1).repeat(1, self.pred_len, 1)
        return pred