import torch
import torch.nn as nn
import torch.fft as fft


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block, splitting a series into trend and seasonal components.
    Uses moving average (via 1D avg pool) with padding to keep length.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        # move channels to second dimension for pooling: (B, C, T)
        x_permuted = x.permute(0, 2, 1)
        # pad on time dimension
        x_padded = nn.functional.pad(x_permuted, (self.padding, self.padding), mode="reflect")
        # apply avg pool
        trend = nn.functional.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        # trend: (B, C, T)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal


class AutoCorrelationLayer(nn.Module):
    """
    Autoformer auto-correlation layer: computes autocorrelation via FFT,
    then aggregates subseries by time-delay aggregation (TDA) using the top-k lags.
    """

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        returns: aggregated: (B, T, C)
        """
        B, T, C = x.shape

        # 1) FFT along time dimension
        # Compute real FFT (frequency domain)
        freq = fft.rfft(x, dim=1)  # (B, T_freq, C), complex

        # 2) Power Spectral Density (auto-spectrum)
        ps = (freq * torch.conj(freq)).real  # (B, T_freq, C)

        # 3) Inverse FFT to get autocorrelation (Wiener-Khinchin theorem)
        # Use symmetric length = 2*T - 2 to capture full correlation
        corr = fft.irfft(ps, n=2 * (T - 1), dim=1)  # (B, 2T-2, C)

        # 4) Keep only non-negative lags (second half)
        corr = corr[:, T - 1 :, :]  # (B, T, C)

        # 5) Average across batch to get mean correlation per channel
        corr_mean = corr.mean(dim=0)  # (T, C)

        # 6) Select top-k lags per channel
        # We exclude lag 0? Paper sometimes excludes, but here we include for simplicity
        topk_vals, topk_idx = torch.topk(corr_mean, self.top_k, dim=0)  # (top_k, C)

        # 7) Compute weights via softmax over top-k values
        weights = torch.softmax(topk_vals, dim=0)  # (top_k, C)

        # 8) Time-delay aggregation: roll + weight
        agg = torch.zeros_like(x)  # (B, T, C)
        for i in range(self.top_k):
            lag = topk_idx[i]  # (C,)
            w = weights[i]     # (C,)
            for c in range(C):
                # roll the time series of channel c by lag[c]
                shift = int(lag[c].item())
                rolled = torch.roll(x[:, :, c], shifts=shift, dims=1)
                agg[:, :, c] += w[c] * rolled

        return agg


class AutoformerEncoderLayer(nn.Module):
    """
    One layer of Autoformer encoder: decomposition + auto-correlation + feedforward + residuals
    """
    def __init__(self, d_model: int, top_k: int, ff_dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)  # kernel_size is a hyperparameter
        self.auto_corr = AutoCorrelationLayer(top_k=top_k)

        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        # Decompose
        trend, seasonal = self.decomp(x)

        # Auto-correlation on seasonal component
        ac = self.auto_corr(seasonal)  # (B, T, C) but here C = d_model if projected

        # Project seasonal part
        seasonal_proj = self.proj_in(seasonal)

        # Add correlation
        seasonal_corr = seasonal_proj + ac

        # Residual + norm
        res1 = self.norm1(seasonal_corr + x)

        # Feed forward
        ff_out = self.ff(res1)
        ff_out = self.dropout(ff_out)

        out = self.norm2(res1 + ff_out)

        # Return out + trend (so we can combine trend later)
        return out, trend


class AutoformerDecoderLayer(nn.Module):
    """
    One layer of Autoformer decoder.
    Uses decomposition + auto-correlation + cross-attention + feedforward.
    """
    def __init__(self, d_model: int, n_heads: int, top_k: int, ff_dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.auto_corr = AutoCorrelationLayer(top_k=top_k)

        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        """
        x: (B, T_dec, d_model)
        enc_output: (B, T_enc, d_model)
        """
        # decomposition
        trend_dec, seasonal_dec = self.decomp(x)

        # compute auto-correlation on seasonal part
        ac_dec = self.auto_corr(seasonal_dec)

        # project seasonal
        seasonal_proj = self.proj_in(seasonal_dec)

        # add self correlation
        seasonal_self = seasonal_proj + ac_dec

        # cross attention with encoder output
        attn_out, _ = self.cross_attn(seasonal_self, enc_output, enc_output)

        # residual + norm
        res1 = self.norm1(attn_out + seasonal_self)

        # feedforward
        ff_out = self.ff(res1)
        ff_out = self.dropout(ff_out)
        res2 = self.norm2(res1 + ff_out)

        # combine with trend
        out = self.norm3(res2 + trend_dec)

        return out


class Autoformer(nn.Module):
    """
    Full Autoformer model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_length: int,
        pred_len: int,
        d_model: int = 128,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        top_k: int = 5,
        n_heads: int = 8,
        ff_dim: int = 256,
        kernel_size: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, top_k, ff_dim, kernel_size, dropout) for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, top_k, ff_dim, kernel_size, dropout) for _ in range(n_decoder_layers)
        ])

        # decoder "start token": learnable parameter
        self.start_token = nn.Parameter(torch.zeros(1, pred_len, d_model))

    def forward(self, x):
        """
        x: (B, T_in, input_dim)
        returns:
            preds: (B, T_out, output_dim)
        """
        B, T, C = x.shape
        assert T == self.input_length, "Input length mismatch"

        # 1) input projection
        enc = self.input_proj(x)  # (B, T, d_model)

        # 2) encoder
        trend_sum = torch.zeros_like(enc)
        seasonal = enc
        for layer in self.encoder_layers:
            seasonal, trend = layer(seasonal)
            trend_sum = trend_sum + trend

        # 3) prepare decoder input
        # use learned start token
        dec_in = self.start_token.repeat(B, 1, 1)  # (B, pred_length, d_model)

        # 4) decoder
        for layer in self.decoder_layers:
            dec_in = layer(dec_in, seasonal)

        # 5) project output
        out = self.output_proj(dec_in)  # (B, pred_length, output_dim)

        return out
