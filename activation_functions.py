import numpy as np
from nptyping import NDArray


def sigmoid(x: NDArray[np.float64]) -> np.float64:
    """
    Squashes the given input quantity into the [0,1] range.
    """
    return 1 / (1 + np.exp(-x))