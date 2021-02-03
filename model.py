from typing import Generator

import numpy as np
from nptyping import NDArray

import activation_functions
import helper_functions
import loss_functions
import metrics


GLOBAL_SEED = 42


class Model:
    def __init__(self):
        pass

    def forward(self):
        pass

    def compute_loss(
        self, y_true: NDArray[int], y_hat: NDArray[np.float64]
    ) -> np.float64:
        """
        Computes the value of the cost function for the given parameters.
        y_true: [N_data]
        y_hat: [N_data]
        """
        return loss_functions.cost_function(
            y_true, y_hat, loss=loss_functions.binary_cross_entropy_loss
        )

    def backward(self):
        pass

    def step(
        self, x: NDArray[int], y_true: NDArray[int]
    ) -> (np.float64, NDArray[np.float64]):
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
        return loss, y_hat

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
        return np.array(preds), np.array(proba)

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
        self, features: NDArray[int], labels: NDArray[int], seed: int
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
        choose_best_weights: bool = True,
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
                Compute our train accuracy & add to list of train accuracies.
                === Validation ===
                [No need to shuffle since we aren't learning here]
                Get predictions on the validation data.
                Compute our validation accuracy & add to list of val accuracies.
        """
        training_loss = []
        validation_loss = []
        training_accuracy = []
        validation_accuracy = []
        best_validation_accuracy = 0
        best_weights = None
        best_bias = 0
        best_epoch = -1
        for epoch in range(epochs):
            # === Training ===
            x_train, y_train = self.shuffle_features_and_labels_together(
                x_train, y_train, GLOBAL_SEED
            )
            x_batches = self.make_batches(x_train, batch_size)
            y_batches = self.make_batches(y_train, batch_size)
            loss = 0
            predictions = []
            for x, y in zip(x_batches, y_batches):
                batch_loss, preds = self.step(x, y)
                loss += batch_loss
                predictions.extend(np.round(preds).astype(int))
            train_loss = loss / (len(x_train) / batch_size)
            training_loss.append(train_loss)
            train_acc = metrics.accuracy(y_train, predictions)
            training_accuracy.append(train_acc)
            # === Validation ===
            predictions, y_hat = self.predict(x_valid)
            valid_loss = self.compute_loss(y_valid, y_hat)
            validation_loss.append(valid_loss)
            valid_acc = metrics.accuracy(y_valid, predictions)
            validation_accuracy.append(valid_acc)
            print(
                f"Epoch {epoch+1}\t| Train Loss: {train_loss}\t| Valid Loss: {valid_loss}\t| Train Acc: {train_acc}\t| Valid Acc: {valid_acc}"
            )
            if choose_best_weights and valid_acc > best_validation_accuracy:
                best_epoch = epoch
                best_validation_accuracy = valid_acc
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
        if choose_best_weights:
            self.weights = best_weights
            self.bias = best_bias
            print(
                f"Best Epoch is {best_epoch+1} with a Validation Accuracy of {best_validation_accuracy}. Using best epoch's weights and bias for the model."
            )
        return training_loss, validation_loss, training_accuracy, validation_accuracy
