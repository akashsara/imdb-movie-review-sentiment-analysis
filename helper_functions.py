import numpy as np
from nptyping import NDArray


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


def initialize_bias(shape: tuple = (1,), value: float = 0.0) -> NDArray[np.float64]:
    """
    Returns the bias of the required shape.
    """
    return np.full(shape=shape, fill_value=value, dtype=np.float64)