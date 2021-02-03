from typing import Generator

import numpy as np
from nptyping import NDArray

import activation_functions
import helper_functions
import loss_functions
import metrics
from model import Model


class LogisticRegression(Model):
    def __init__(
        self,
        input_shape: tuple,
        weights_mode: tuple,
        bias: float = 0,
        learning_rate: float = 0.1,
    ):
        """
        input_shape: [N_data, N_features]
        weights_mode: (weight_init_scheme, {"param": "value"})
        Initialize our weights & biases.
        Set learning rate.
        """
        self.weights = helper_functions.initialize_weights(
            input_shape[1], mode=weights_mode
        )
        self.bias = helper_functions.initialize_bias((1,), bias)
        self.learning_rate = learning_rate
        super().__init__()

    def forward(self, x: NDArray[int]) -> NDArray[np.float64]:
        """
        Our forward pass.
        x: [N_data, N_features]
        Pseudocode:
            z = x.W + b
            sigmoid(z)
            Return predictions
        """
        return activation_functions.sigmoid(np.dot(x, self.weights) + self.bias)

    def backward(
        self, x: NDArray[int], y_true: NDArray[int], y_hat: NDArray[np.float64]
    ):
        """
        Our backward pass.
        x: [N_data, N_features]
        y_true: [N_data]
        y_hat: [N_data]
        Pseudocode:
            Compute gradients w.r.t W and b
            Update parameters
        """
        error = y_hat - y_true
        dW = 1 / len(y_true) * np.dot(x.T, error)
        dB = 1 / len(y_true) * np.sum(error)
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB
