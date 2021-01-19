import numpy as np
from nptyping import NDArray


def cost_function(
    y_true: NDArray[int],
    y_hat: NDArray[np.float64],
    loss=binary_cross_entropy_loss,
) -> np.float64:
    """
    The cost function is simply the average of all the losses.
    Given our y_true and y_hat, we calculate the loss for each datapoint.
    We then take the average of this loss and return it.
    This function is independent of the number of data points.
    So it can be used for any batch size.
    """
    losses = loss(y_true, y_hat)
    return np.sum(losses) / len(losses)


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