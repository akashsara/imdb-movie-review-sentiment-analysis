import numpy as np
from nptyping import NDArray


def accuracy(y: NDArray[int], y_hat: NDArray[int]) -> np.float64:
    """
    Computes the accuracy of the predictions.
    Formula: Number of Correct Predictions / Number of Predictions
    """
    if len(y) == len(y_hat):
        return np.sum(y == y_hat) / len(y)
    raise ValueError(f"y and y_hat are not the same shape: {y.shape, y_hat.shape}")