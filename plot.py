import matplotlib.pyplot as plt
from typing import List


def plot_per_epoch(train: List[float], val: List[float], metric: str, save_as: str):
    """
    Plots some training and validation metric with respect to epochs (time).
    Saves the plot with the filename specified in save_as.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(train)
    plt.plot(val)
    plt.legend([f"training {metric}", f"validation {metric}"], loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.savefig(save_as)