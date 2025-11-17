import torch
import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

    def visualize_single_channel(self,
                                 x: Union[pd.DataFrame | np.ndarray | torch.Tensor],
                                 channel: int = 0,
                                 sample: int = 0,
                                 channel_first: bool = True,
                                 title: str = "Data Plot",
                                 xlabel: str = "X-axis",
                                 ylabel: str = "Y-axis"
                                 ):
        if type(channel)==int and type(x) == pd.DataFrame:
            x = x.to_numpy()
        if len(x.shape) == 2:
            if channel_first:
                channel = min(channel, x.shape[0] - 1)
                x = x[channel]
            else:
                channel = min(channel, x.shape[1] - 1)
                x = x[:, channel]
        elif len(x.shape) == 3:
            # leading batchsize dimension
            if channel_first:
                channel = min(channel, x.shape[1] - 1)
                x = x[sample, channel]
            else:
                channel = min(channel, x.shape[2] - 1)
                x = x[sample, :, channel]

        plt.figure(figsize=(10, 6))
        plt.plot(x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()


    def visualize_multiple_channels(self,
                                    x: Union[pd.DataFrame | np.ndarray | torch.Tensor],
                                    channels: List[int|str] = None,
                                    sample: int = 0,
                                    channel_first: bool = True,
                                    title: str = "Data Plot",
                                    xlabel: str = "X-axis",
                                    ylabel: str = "Y-axis"
                                    ):
        plt.figure(figsize=(10, 6))
        if type(channels[0])==int and type(x) == pd.DataFrame:
            x = x.to_numpy()

        for channel in channels:
            if len(x.shape) == 2:
                if channel_first:
                    channel = min(channel, x.shape[0] - 1)
                    x_plot = x[channel]
                else:
                    channel = min(channel, x.shape[1] - 1)
                    x_plot = x[:, channel]
            elif len(x.shape) == 3:
                # leading batchsize dimension
                if channel_first:
                    channel = min(channel, x.shape[1] - 1)
                    x_plot = x[sample, channel]
                else:
                    channel = min(channel, x.shape[2] - 1)
                    x_plot = x[sample, :, channel]

            plt.plot(x_plot)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()