from logistic_regression import LogisticRegression
import plot
import metrics
import sys
import pickle
import os
from datetime import datetime

# Data Paths
train_data_path = sys.argv[1]  # data/training_data.pkl
valid_data_path = sys.argv[2]  # data/validation_data.pkl
test_data_path = sys.argv[3]  # data/testing_data.pkl

# Experiment Parameters
weights = ("uniform", {"low": -0.5, "high": 0.5})
bias = 0.0
learning_rate = 0.1
epochs = 300
batch_size = 20

with open(train_data_path, "rb") as fp:
    x_train, y_train = pickle.load(fp)
with open(valid_data_path, "rb") as fp:
    x_valid, y_valid = pickle.load(fp)

model = LogisticRegression(
    input_shape=x_train.shape,
    weights_mode=weights,
    bias=bias,
    learning_rate=learning_rate,
)

# Training & Validation
train_loss, valid_loss, train_accuracy, valid_accuracy = model.train(
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    epochs=epochs,
    batch_size=batch_size,
)

# Create Output Folder
if not os.path.exists("outputs"):
    os.mkdir("outputs")

output_dir = os.path.join("outputs", str(int(datetime.utcnow().timestamp())))
os.mkdir(output_dir)

# Plot Graphs of Loss & Accuracy
plot.plot_per_epoch(
    train_accuracy, valid_accuracy, "accuracy", os.path.join(output_dir, "accuracy.png")
)
plot.plot_per_epoch(
    train_loss, valid_loss, "loss", os.path.join(output_dir, "loss.png")
)

# Testing
with open(test_data_path, "rb") as fp:
    x_test, y_test = pickle.load(fp)
y_pred, _ = model.predict(x_test, batch_size=batch_size)
accuracy = metrics.accuracy(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy}")