from typing import Generator

import numpy as np
from nptyping import NDArray

import activation_functions
import helper_functions
import loss_functions
import metrics
from model import Model

class NeuralNetwork(Model):
    def __init__(
        self,
        input_shape: tuple,
        weights_mode: tuple,
        bias: float = 0,
        learning_rate: float = 0.1,
        hidden_dimension: int = 200,
    ):
        """
        input_shape: [N_data, N_features]
        hidden_dimension: Number of nodes in the hidden layer
        weights_mode: (weight_init_scheme, {"param": "value"})
        Initialize our weights & bia
        # TODO: Look at making it more generic - supporting multiple layers etc.
        """
        hidden_weights = helper_functions.initialize_weights(
            (input_shape[1], hidden_dimension), mode=weights_mode
        )
        hidden_bias = helper_functions.initialize_bias((1, hidden_dimension), bias)
        output_weights = helper_functions.initialize_weights(
            (hidden_dimension, 1), mode=weights_mode
        )
        output_bias = helper_functions.initialize_bias((1,), bias)
        self.weights = {"W0": hidden_weights, "W1": output_weights}
        self.bias = {"b0": hidden_bias, "b1": output_bias}
        self.cache = {}
        self.learning_rate = learning_rate
        super().__init__()

    def forward(self, x: NDArray[int]) -> NDArray[np.float64]:
        """
        Our forward pass.
        x: [N_data, N_features]
        Pseudocode:
            z0 = x.W0 + b0
            a0 = sigmoid(z0)
            z1 = a0.W1 + b1
            a1 = sigmoid(z1)
            Return predictions (a1)
        """
        a0 = activation_functions.sigmoid(
            np.dot(x, self.weights["W0"]) + self.bias["b0"]
        )
        a1 = activation_functions.sigmoid(
            np.dot(a0, self.weights["W1"]) + self.bias["b1"]
        )
        self.cache = {"a0": a0, "a1": a1}
        return a1

    def backward(
        self, x: NDArray[int], y_true: NDArray[int], y_hat: NDArray[np.float64]
    ):
        """
        Our backward pass.
        x: [N_data, N_features]
        y_true: [N_data, 1]
        y_hat = a1: [N_data, 1]
        a0: [N_data, hidden_dimension]
        W1: [hidden_dimension, 1]
        W0: [N_features, hidden_dimension]
        Pseudocode:
            Compute gradients w.r.t W and b
            Update parameters
        dW1 = 1/batch_size * dot(a0.T, error)
        db1 = 1/batch_size * sum(error)
        dh  = dot(error, w1.T) * a0 * (1 -a0)
        dW0 = 1/batch_size * dot(x.T, dh)
        db0 = 1/batch_size * sum(dh, axis=0)
        """
        error = y_hat - y_true
        dW1 = 1 / len(y_true) * np.dot(self.cache["a0"].T, error)
        db1 = 1 / len(y_true) * np.sum(error)
        dh = (
            np.dot(error, self.weights["W1"].T)
            * self.cache["a0"]
            * (1 - self.cache["a0"])
        )
        dW0 = 1 / len(y_true) * np.dot(x.T, dh)
        db0 = 1 / len(y_true) * np.sum(dh, axis=0)
        self.weights["W1"] -= self.learning_rate * dW1
        self.bias["b1"] -= self.learning_rate * db1
        self.weights["W0"] -= self.learning_rate * dW0
        self.bias["b0"] -= self.learning_rate * db0
        self.cache = {}
