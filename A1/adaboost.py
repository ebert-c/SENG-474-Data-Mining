from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from utils import plot_test_and_train_error, print_best_values


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True)
    vary_iterations(X_train, X_test, y_train, y_test)
    vary_training_set_size(X, y)
    vary_maximum_depth(X_train, X_test, y_train, y_test)


def vary_maximum_depth(X_train, X_test, y_train, y_test):
    train_error = []
    test_error = []
    depth = []
    for i in range(1, 100, 10):
        boost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i))
        boost_classifier.fit(X_train, y_train)
        train_error.append(1 - boost_classifier.score(X_train, y_train))
        test_error.append(1 - boost_classifier.score(X_test, y_test))
        depth.append(i)
    print_best_values(test_error, train_error, depth, "DEPTH", "Depth", "adaboost_stats")
    plot_test_and_train_error(test_error, train_error, depth, "Maximum Estimator Depth", 1)


def vary_iterations(X_train, X_test, y_train, y_test):
    train_error = []
    test_error = []
    iterations = []
    for i in range(10, 140, 10):
        boost_classifier = AdaBoostClassifier(n_estimators=i)
        boost_classifier.fit(X_train, y_train)
        train_error.append(1 - boost_classifier.score(X_train, y_train))
        test_error.append(1 - boost_classifier.score(X_test, y_test))
        iterations.append(i)
    print_best_values(test_error, train_error, iterations, "ITERATIONS", "Iterations", "adaboost_stats")
    plot_test_and_train_error(test_error, train_error, iterations, "Number of Iterations", 1)


def vary_training_set_size(X, y):
    train_error = []
    test_error = []
    train_sizes = []
    for i in range(1, 10):
        train_size = round(0.1 * i, 2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
        boost_classifier = AdaBoostClassifier()
        boost_classifier.fit(X_train, y_train)
        train_error.append(1 - boost_classifier.score(X_train, y_train))
        test_error.append(1 - boost_classifier.score(X_test, y_test))
        train_sizes.append(train_size)
    print_best_values(test_error, train_error, train_sizes, "TRAINING SET SIZE", "Training Set Size", "adaboost_stats")
    plot_test_and_train_error(test_error, train_error, train_sizes, "Size of Training Data", 4)


if __name__ == "__main__":
    main()
