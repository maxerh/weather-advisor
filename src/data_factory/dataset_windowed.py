import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    """
    Converts a multivariate time series into sliding windows for forecasting.

    This dataset supports:
    - Multivariate input (N features)
    - Multi-step predictions (prediction_length > 1)
    - Optional selection of specific target columns
    - Clean conversion to float32 tensors

    Parameters
    ----------
    data : pandas.DataFrame or np.ndarray
        Input time series data where each row is a timestamp and each column is a feature.
    input_length : int
        Number of past time steps to use as model input (X).
    prediction_length : int
        Number of future time steps to predict (Y).
    target_cols : list[str] or list[int], optional
        Columns to predict. If None (default), all columns are predicted.
    """

    def __init__(self, data, input_length: int, prediction_length: int, target_cols=None):
        super().__init__()

        # Convert to numpy
        if hasattr(data, "values"):  # pandas DataFrame
            self.column_names = data.columns.tolist()[1:]
            data = data.values
        else:
            self.column_names = range(data.shape[1] - 1)

        self.data = np.asarray(data[:,1:], dtype=np.float32)
        self.timestamps = data[:, 0]  # assuming first row is timestamps or headers

        self.input_length = input_length
        self.pred_len = prediction_length

        # Determine target columns (default: all columns)
        if target_cols is None:
            self.target_cols = list(range(self.data.shape[1]))
        else:
            self.target_cols = [
                self.column_names.index(col) if isinstance(col, str) else col for col in target_cols
            ]

        self.num_samples = len(self.data) - self.input_length - self.pred_len

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough data points ({len(self.data)}) for "
                f"{input_length=} and {prediction_length=}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            X : tensor of shape (input_length, num_features)
            Y : tensor of shape (prediction_length, num_target_features)
        """
        start_x = idx
        end_x = idx + self.input_length

        start_y = end_x
        end_y = end_x + self.pred_len

        X = self.data[start_x:end_x]  # (input_length, num_features)
        Y = self.data[start_y:end_y][:, self.target_cols]  # (pred_len, num_target_features)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)
        )


def create_datasets_from_splits(df_train, df_val, df_test,
                                input_length, prediction_length,
                                target_cols=None):
    """
    Convenience utility to wrap all splits into WindowedDataset objects.

    Example:
        train_ds, val_ds, test_ds = create_datasets_from_splits(
            df_train_norm, df_val_norm, df_test_norm,
            input_length=168,
            prediction_length=24
        )
    """
    train_ds = WindowedDataset(df_train, input_length, prediction_length, target_cols)
    val_ds = WindowedDataset(df_val, input_length, prediction_length, target_cols)
    test_ds = WindowedDataset(df_test, input_length, prediction_length, target_cols)

    return train_ds, val_ds, test_ds