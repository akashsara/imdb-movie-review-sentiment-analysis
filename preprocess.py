import os
import sys
import pickle
from collections import Counter

import numpy as np

train_dir = sys.argv[1]  # aclImdb/train/
test_dir = sys.argv[2]  # aclImdb/test/
validation_split = float(sys.argv[3])  # 0.2
vocabulary_size = int(sys.argv[4])  # 2000
GLOBAL_SEED = 42


def preprocess_text(text):
    return text.strip().lower().split()


def read_files_in_directory(directory, preprocess_func):
    data = []
    for i, file_ in enumerate(os.listdir(directory)):
        with open(os.path.join(directory, file_), "r", encoding="utf8") as fp:
            text = fp.read()
            data.append(preprocess_func(text))
        if i % 1000 == 0:
            print(f"{i} files read from {directory}")
    print("---" * 20)
    # We return it as an np array for easier & vectorized manipulation
    return np.array(data, dtype="object")


def shuffle_features_and_labels_together(features, labels, seed=GLOBAL_SEED):
    np.random.seed(seed)
    np.random.shuffle(features)
    np.random.seed(seed)
    np.random.shuffle(labels)
    return features, labels


def create_feature_vector(data, word2id):
    feature_vector = np.zeros((len(data), len(word2id)))
    vocab = set(word2id.keys())
    for i, sentence in enumerate(data):
        index = [word2id[word] for word in set(sentence).intersection(vocab)]
        feature_vector[i][index] = 1
    return feature_vector


# Training & Validation Data
train_pos = read_files_in_directory(
    directory=os.path.join(train_dir, "pos"), preprocess_func=preprocess_text
)
train_neg = read_files_in_directory(
    directory=os.path.join(train_dir, "neg"), preprocess_func=preprocess_text
)

# Testing Data
test_pos = read_files_in_directory(
    directory=os.path.join(test_dir, "pos"), preprocess_func=preprocess_text
)
test_neg = read_files_in_directory(
    directory=os.path.join(test_dir, "neg"), preprocess_func=preprocess_text
)

# Shuffle Training Data
np.random.shuffle(train_pos)
np.random.shuffle(train_neg)

# Split Training Data into Training & Validation Data
validation_size = np.round(train_pos.shape[0] * validation_split).astype(int)
train_size = train_pos.shape[0] - validation_size
print(f"Training Data Size: {train_size * 2}")
print(f"Validation Data Size: {validation_size * 2}")
valid_pos = train_pos[:validation_size]
valid_neg = train_neg[:validation_size]
train_pos = train_pos[validation_size:]
train_neg = train_neg[validation_size:]

# Combine Training Data, Add Labels & Mix Training Data
x_train = train_pos + train_neg
y_train = [1] * len(train_pos) + [0] * len(train_neg)
x_train, y_train = shuffle_features_and_labels_together(x_train, y_train)

# Combine Validation Data, Add Labels & Mix Validation Data
# Note:
# Its not necessary to mix the validation data since we aren't learning from it.
x_valid = valid_pos + valid_neg
y_valid = [1] * len(valid_pos) + [0] * len(valid_neg)
x_valid, y_valid = shuffle_features_and_labels_together(x_valid, y_valid)

# Combine Test Data, Add Labels & Mix Test Data
# Not necessary to mix the data here either.
x_test = test_pos + test_neg
y_test = [1] * len(test_pos) + [0] * len(test_neg)
x_test, y_test = shuffle_features_and_labels_together(x_test, y_test)

# Construct Vocabulary from Training Data
word_counts = Counter([word for sentence in x_train for word in sentence])
word2id = {
    word: i for i, (word, _) in enumerate(word_counts.most_common(vocabulary_size))
}
id2word = {i: word for word, i in word2id.items()}

# Save vocabulary info
with open(os.path.join("data", "vocabulary.pkl"), "wb") as fp:
    pickle.dump((word2id, id2word, word_counts), fp)
print("Vocabulary saved!")

# Make Features
x_train = create_feature_vector(x_train, word2id)
x_valid = create_feature_vector(x_valid, word2id)
x_test = create_feature_vector(x_test, word2id)

with open(os.path.join("data", "training_data.pkl"), "wb") as fp:
    pickle.dump((x_train, y_train), fp)
with open(os.path.join("data", "validation_data.pkl"), "wb") as fp:
    pickle.dump((x_valid, y_valid), fp)
with open(os.path.join("data", "testing_data.pkl"), "wb") as fp:
    pickle.dump((x_test, y_test), fp)
print("Data saved!")
