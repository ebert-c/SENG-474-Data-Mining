from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    alg = DecisionTreeClassifier()
    k_fold(alg)


def k_fold(algorithm):
    K = 5

    errors = []
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    x_folds = []
    y_folds = []
    fold_len = int(len(X_train) / K)

    for i in range(0, len(X_train), fold_len):
        x_folds.append(X_train[i:i + fold_len])
        y_folds.append(y_train[i:i + fold_len])

    for k in range(K):
        fold_x = []
        fold_y = []
        for i in range(K):
            if i == k:
                continue
            fold_x += x_folds[i]
            fold_y += y_folds[i]
        algorithm.fit(fold_x, fold_y)
        errors.append(1 - algorithm.score(X_train, y_train))
    risk = sum(errors) / K
    print(risk)


if __name__ == "__main__":
    main()
