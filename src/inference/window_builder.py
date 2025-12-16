import torch

def build_last_window(df, input_length):
    """
    df: transformed dataframe (after fg.transform)
    Returns tensor of shape [1, input_length, num_features]
    """
    if len(df) < input_length:
        raise ValueError("Not enough data for inference window")

    data = df.iloc[-input_length:].values
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    return x