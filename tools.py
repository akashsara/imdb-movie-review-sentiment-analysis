# TODO: Split functions into relevant files. Figure out our folder structure
import numpy as np
from nptyping import NDArray


def sigmoid(x: int) -> np.float64:
    """
    Squashes the given input quantity into the [0,1] range.
    """
    return 1 / (1 + np.exp(-x))


def initialize_weights(
    shape: tuple, mode: str = "uniform", low: float = -0.5, high: float = 0.5
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
    if mode == "uniform":
        return np.random.uniform(low=low, high=high, size=shape)
    elif mode == "zero":
        return np.zeros(shape=shape)
    else:
        raise ValueError("Invalid weight initialization mode specified")


def initialize_bias(value: float = 0.0) -> float:
    """
    Seems like a useless function but I'd rather have this called explicitly.
    """
    return value


def binary_cross_entropy_loss(
    y: NDArray[int], y_hat: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Binary CrossEntropy Loss is defined as:
        loss = -[y * log(yhat) + (1 - y) * log(1 - yhat)]
        When y = 1:
            loss = -log(yhat)
        When y = 0:
            loss = -log(1 - yhat)
    """
    return np.where(y == 1, -np.log(y_hat), -np.log(1 - y_hat))


def cost_function(
    y_true: NDArray[int],
    y_hat: NDArray[np.float64],
    loss_function=binary_cross_entropy_loss,
) -> np.float64:
    """
    The cost function is simply the average of all the losses.
    Given our y_true and y_hat, we calculate the loss for each datapoint.
    We then take the average of this loss and return it.
    This function is independent of the number of data points.
    So it can be used for any batch size.
    """
    losses = loss_function(y_true, y_hat)
    return np.sum(losses) / len(losses)