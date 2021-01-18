import numpy as np
from collections import Counter

train_dir = ""
test_dir = ""
validation_split = 0.2
vocabulary_size = 2000
GLOBAL_SEED = 42


def preprocess_text(text):
    return text.strip().lower().split()


def read_files_in_directory(directory, preprocess_func):
    data = []
    for file_ in os.listdir(directory):
        with open(os.path.join(directory, file_), "r", encoding="utf8") as fp:
            text = fp.read()
            data.append(preprocess_func(text))
    return np.array(data)


def shuffle_features_and_labels_together(features, labels, seed=GLOBAL_SEED):
    np.random.seed(seed)
    np.random.shuffle(features)
    np.random.seed(seed)
    np.random.shuffle(labels)
    return features, labels


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
valid_pos, train_pos = np.split(train_pos, [validation_size, train_size])
valid_neg, train_neg = np.split(train_neg, [validation_size, train_size])

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
vocabulary = [word for word, counts in word_counts.most_common(vocabulary_size)]