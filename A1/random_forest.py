from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # vary_estimators(X_train, y_train, X_test, y_test)
    vary_feature_number(X_train, y_train, X_test, y_test)


def plot_test_and_train_error(test_error, train_error, x_axis, xlabel):
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.legend()
    plt.show()


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
    plot_test_and_train_error(test_error, train_error, feature_nums, "Number of Features")



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
    plot_test_and_train_error(test_error, train_error, estimators, "Number of Estimators")


if __name__ == "__main__":
    main()
