import A2.utils.mnist_reader as mnist_reader
import random

P = 0.2


def load_data():
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = [[], [], [], []]
    X_train, y_train = mnist_reader.load_mnist("fashion", kind="train")
    X_test, y_test = mnist_reader.load_mnist("fashion", kind="t10k")

    for i in range(len(X_train)):
        if y_train[i] in [5, 7]:
            y_noisy = 0 if y_train[i] == 5 else 1
            x_rescaled = [j / 255 for j in X_train[i]]
            X_train_filtered.append(x_rescaled)
            y_noisy = random.choices([int(not y_noisy), y_noisy], weights=[P, 1 - P])[0]
            y_train_filtered.append(y_noisy)

    for i in range(len(X_test)):
        if y_test[i] in [5, 7]:
            y_filtered = 0 if y_test[i] == 5 else 1
            X_test_filtered.append(X_test[i])
            y_test_filtered.append(y_filtered)

    return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered
