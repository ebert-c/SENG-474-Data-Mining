from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True)
    # vary_estimators(X_train, y_train, X_test, y_test)
    # vary_feature_number(X_train, y_train, X_test, y_test)
    vary_training_set_size(X, y)


def plot_test_and_train_error(test_error, train_error, x_axis, xlabel, i):
    plt.figure(i)
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def vary_training_set_size(X, y):
    train_error = []
    test_error = []
    train_sizes = []
    for i in range(1, 10):
        train_size = 0.1 * i
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        train_error.append(1 - rfc.score(X_train, y_train))
        test_error.append(1 - rfc.score(X_test, y_test))
        train_sizes.append(train_size)
    plot_test_and_train_error(test_error, train_error, train_sizes, "Size of Training Data", 4)


def vary_feature_number(X_train, y_train, X_test, y_test):
    train_error = []
    test_error = []
    feature_nums = []
    min_error = 1
    optimal_feature_nums = 0

    for i in range(1, 58):
        rfc = RandomForestClassifier(max_features=i)
        rfc.fit(X_train, y_train)
        training_error = 1 - rfc.score(X_train, y_train)
        testing_error = 1 - rfc.score(X_test, y_test)
        train_error.append(training_error)
        test_error.append(testing_error)
        feature_nums.append(i)
        if testing_error < min_error:
            min_error = testing_error
            optimal_feature_nums = i

    print(min_error)
    print(optimal_feature_nums)
    plot_test_and_train_error(test_error, train_error, feature_nums, "Number of Features", 1)


def vary_estimators(X_train, y_train, X_test, y_test):
    train_error = []
    test_error = []
    estimators = []
    min_error = 1
    optimal_n = 0

    for i in range(50, 150):
        rfc = RandomForestClassifier(n_estimators=i)
        rfc.fit(X_train, y_train)
        training_error = 1 - rfc.score(X_train, y_train)
        testing_error = 1 - rfc.score(X_test, y_test)
        train_error.append(training_error)
        test_error.append(testing_error)
        estimators.append(i)
        if testing_error < min_error:
            min_error = testing_error
            optimal_n = i

    print(min_error)
    print(optimal_n)
    plot_test_and_train_error(test_error, train_error, estimators, "Number of Estimators", 2)


if __name__ == "__main__":
    main()
