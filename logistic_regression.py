from typing import Generator

import numpy as np
from nptyping import NDArray

import loss_functions
import helper_functions
import activation_functions
import metrics

# TODO: Testing

GLOBAL_SEED = 42


class LogisticRegression:
    def __init__(
        self,
        input_shape: tuple,
        weights_mode: tuple[str, dict],
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
        self.bias = helper_functions.initialize_bias(bias)
        self.learning_rate = learning_rate

    def forward(self, x: NDArray[int]) -> NDArray[np.float64]:
        """
        Our forward pass.
        x: [N_data, N_features]
        Pseudocode:
            z = x.W + b
            sigmoid(z)
            Return predictions
        """
        return activation_functions.sigmoid(np.dot(x, self.weights) + bias)

    def compute_loss(y_true: NDArray[int], y_hat: NDArray[np.float64]) -> np.float64:
        """
        Computes the value of the cost function for the given parameters.
        y_true: [N_data]
        y_hat: [N_data] 
        """
        return loss_functions.cost_function(
            y_true, y_hat, loss=loss_functions.binary_cross_entropy_loss
        )

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
        error = y_hat - y
        dW = 1 / len(y_true) * np.dot(x.T, error)
        dB = 1 / len(y_true) * np.sum(error)
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB

    def step(self, x: NDArray[int], y_true: NDArray[int]) -> np.float64:
        """
        One step of our optimization.
        x: [N_data, N_features]
        y_true: [N_data]
        Pseudocode:
            forward pass on a batch of data.
            calculate loss based on predictions.
            backward pass (updating our weights & biases)
            return loss of batch.
        Remember, 1 step /= 1 epoch.
        1 epoch will have len(data) / batch_size steps.
        """
        y_hat = self.forward(x)
        loss = self.compute_loss(y_true, y_hat)
        self.backward(x, y_true, y_hat)
        return loss

    def predict(
        self, x: NDArray[int], batch_size: int = 20
    ) -> (NDArray[int], NDArray[np.float64]):
        """
        This is the function we'll use for making predictions.
        x: [N_data, N_features]
        Pseudocode:
            Predict on data 
            Returns (preds, proba)
        Where logits are the raw outputs of the model and
        predictions are the rounded integer predictions.
        Note:
        Unlike in softmax,
        we can't consider the output of sigmoid as a true confidence.
        """
        batches = self.make_batches(x, batch_size)
        preds = []
        proba = []
        for batch in batches:
            predictions = self.forward(batch)
            proba.extend(predictions)
            preds.extend(np.round(predictions).astype(int))
        return np.array(preds), proba

    def make_batches(self, x: NDArray, batch_size: int):
        """
        Simple function that batches data into chunks of the given size.
        Note that we return a generator here, not a full batch of data.
        x: [N_data, N_features]
        """

        def batch_generator(
            x: NDArray, batch_size: int
        ) -> Generator[NDArray, None, None]:
            for i in range(0, len(x), batch_size):
                yield x[i : i + batch_size]

        return batch_generator(x, batch_size)

    def shuffle_features_and_labels_together(
        features: NDArray[int], labels: NDArray[int], seed: int
    ) -> (NDArray[int], NDArray[int]):
        """
        Shuffles the given NDArrays on the first axis alone.
        So we are only randomizing the order of the features & labels.
        Note that due to us setting the seed explicitly, the mapping
        between the features and the labels in maintained. 
        """
        np.random.seed(seed)
        np.random.shuffle(features)
        np.random.seed(seed)
        np.random.shuffle(labels)
        return features, labels

    def train(
        self,
        x_train: NDArray[int],
        y_train: NDArray[int],
        x_valid: NDArray[int],
        y_valid: NDArray[int],
        epochs: int = 300,
        batch_size: int = 20,
    ):
        """
        Our training & validation loop.
        x_train, x_valid: [N_data, N_features]
        y_train, y_valid: [N_data]
        Pseudocode:
            For every epoch:
                === Training ===
                Shuffle data & batch it.
                Set loss = 0
                For every batch:
                    Take one step of the optimizer.
                    loss += this batch's loss
                Divide loss / num_batches
                Append loss to list of training losses.
                === Validation ===
                [No need to shuffle since we aren't learning here]
                Get predictions on the validation data.
                Compute our validation accuracy & add to list of accuracies.
        """
        training_loss = []
        validation_accuracy = []
        for epoch in len(epochs):
            # === Training ===
            x_train, y_train = shuffle_features_and_labels_together(
                x_train, y_train, GLOBAL_SEED
            )
            x_batches = make_batches(x_train, batch_size)
            y_batches = make_batches(y_train, batch_size)
            loss = 0
            for x, y in zip(x_batches, y_batches):
                loss += self.step(x, y)
            training_loss.append(loss / len(batches))
            # === Validation ===
            predictions, _ = self.predict(x_valid)
            accuracy = metrics.accuracy(y_valid, predictions)
            validation_accuracy.append(accuracy)
        return training_loss, validation_accuracy
