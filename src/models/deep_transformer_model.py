import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return x


class DeepTransformerForecastModel(nn.Module):
    """
    High-quality forecasting Transformer with:
      - multi-layer encoder
      - positional encodings
      - multi-step decoder
      - independent predictions for each future timestep

    Input:
        x: (B, T_in, input_dim)

    Output:
        preds: (B, T_out, output_dim)
    """

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
        *args, **kwargs
    ):
        super().__init__()

        self.pred_len = pred_len
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=input_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pred_len * output_dim)
        )

    def forward(self, x):
        """
        x: shape (B, T_in, input_dim)
        """
        x = self.input_projection(x)          # (B, T_in, embed_dim)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)             # (B, T_in, embed_dim)
        last = encoded[:, -1, :]              # (B, embed_dim)
        decoded = self.decoder(last)          # (B, pred_len * output_dim)
        preds = decoded.view(-1, self.pred_len, self.output_dim)

        return preds