# TODO: Split functions into relevant files. Figure out our folder structure
import numpy as np
from nptyping import NDArray


def sigmoid(x: NDArray[np.float64]) -> np.float64:
    """
    Squashes the given input quantity into the [0,1] range.
    """
    return 1 / (1 + np.exp(-x))


def initialize_weights(
    shape: tuple, mode: tuple = ("uniform", {"low": -0.5, "high": 0.5})
) -> NDArray[np.float64]:
    """
    Returns an initialized weight matrix of the required shape.
    Modes currently supported are "uniform" and "zero".
    The assignment only requires uniform initialization.
    But I wanted to see the difference in the results so I've included zero.

    uniform: Samples are drawn from a uniform distribution [low, high)
    This means any value in the distribution is equally likely to be drawn.
    I've set low=-0.5 and high = 0.5 as default values as required.
    However, since they are supplied as arguments, they can be changed.
    """
    kwargs = mode[1]
    mode = mode[0]
    if mode == "uniform":
        return np.random.uniform(size=shape, **kwargs)
    elif mode == "zero":
        return np.zeros(shape=shape)
    else:
        raise ValueError("Invalid weight initialization mode specified")


def initialize_bias(value: float = 0.0) -> float:
    """
    Seems like a useless function but I'd rather have this called explicitly.
    """
    return value