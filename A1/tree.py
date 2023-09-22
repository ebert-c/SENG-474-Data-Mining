from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

BASELINE_CCP = 0.001


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #  vary_ccp_alpha(X_train, y_train, X_test, y_test)
    vary_split_criterion(X_train, y_train, X_test, y_test)


def plot_test_and_train_error(test_error, train_error, x_axis, xlabel):
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def vary_split_criterion(X_train, y_train, X_test, y_test):
    clf_gini = DecisionTreeClassifier()
    clf_entropy = DecisionTreeClassifier(criterion="entropy")
    clf_log_loss = DecisionTreeClassifier(criterion="log_loss")

    criteria = ["Gini", "Entropy", "Log Loss"]
    train_error = []
    test_error = []

    clf_gini.fit(X_train, y_train)
    train_error.append(1 - clf_gini.score(X_train, y_train))
    test_error.append(1 - clf_gini.score(X_test, y_test))

    clf_entropy.fit(X_train, y_train)
    train_error.append(1 - clf_entropy.score(X_train, y_train))
    test_error.append(1 - clf_entropy.score(X_test, y_test))

    clf_log_loss.fit(X_train, y_train)
    train_error.append(1 - clf_log_loss.score(X_train, y_train))
    test_error.append(1 - clf_log_loss.score(X_test, y_test))

    plt.figure(2, label="Test Error")
    plt.bar(criteria, test_error)
    plt.figure(3, label="Training Error")
    plt.bar(criteria, train_error)
    plt.show()


def vary_ccp_alpha(X_train, y_train, X_test, y_test):
    train_error = []
    test_error = []
    ccp_alphas = []
    min_error = 1
    optimal_ccp = 0.0

    for i in range(0, 1000):
        ccp_alpha = i / 1000

        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        training_error = 1 - clf.score(X_train, y_train)
        testing_error = 1 - clf.score(X_test, y_test)
        depth = clf.get_depth()
        if depth <= 1:
            print('this happened')
            print(ccp_alpha)
            break
        train_error.append(training_error)
        test_error.append(testing_error)
        ccp_alphas.append(ccp_alpha)
        if testing_error < min_error:
            min_error = testing_error
            optimal_ccp = ccp_alpha

    print(min_error)
    print(optimal_ccp)
    plot_test_and_train_error(test_error, train_error, ccp_alphas, "CCP Alpha")


if __name__ == "__main__":
    main()
