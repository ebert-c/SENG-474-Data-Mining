from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_test_and_train_error, print_best_values

BASELINE_CCP = 0.01


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    vary_ccp_alpha(X_train, y_train, X_test, y_test)
    vary_split_criterion(X_train, y_train, X_test, y_test)
    vary_training_set_size(X, y)


def vary_training_set_size(X, y):
    train_error = []
    test_error = []
    train_sizes = []
    for i in range(1, 10):
        train_size = round(0.1 * i, 2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
        clf = DecisionTreeClassifier(ccp_alpha=BASELINE_CCP)
        clf.fit(X_train, y_train)
        train_error.append(1 - clf.score(X_train, y_train))
        test_error.append(1 - clf.score(X_test, y_test))
        train_sizes.append(train_size)

    print_best_values(test_error, train_error, train_sizes, "TRAINING SET SIZE", "Training Set Size (Tree)", "tree_stats")
    plot_test_and_train_error(test_error, train_error, train_sizes, "Size of Training Data", 4)


def vary_split_criterion(X_train, y_train, X_test, y_test):
    clf_gini = DecisionTreeClassifier(ccp_alpha=BASELINE_CCP)
    clf_entropy = DecisionTreeClassifier(criterion="entropy", ccp_alpha=BASELINE_CCP)

    stuff = ["Train Error", "Test Error"]
    entropy_error = []
    gini_error = []

    clf_gini.fit(X_train, y_train)
    gini_error.append(1 - clf_gini.score(X_train, y_train))
    gini_error.append(1 - clf_gini.score(X_test, y_test))

    clf_entropy.fit(X_train, y_train)
    entropy_error.append(1 - clf_entropy.score(X_train, y_train))
    entropy_error.append(1 - clf_entropy.score(X_test, y_test))

    X_axis = np.arange(len(stuff))

    plt.figure(2, label="Test Error")
    plt.bar(X_axis - 0.2, gini_error, 0.4, label="Gini")
    plt.bar(X_axis + 0.2, entropy_error, 0.4, label="Entropy")
    plt.ylabel("Error")
    plt.xticks(X_axis, stuff)
    plt.legend()
    plt.savefig("Vary Criteria.svg", format="svg")

    print_best_values(gini_error, entropy_error, stuff, "SPLIT CRITERIA", "Split Criteria", "tree_stats")


def vary_ccp_alpha(X_train, y_train, X_test, y_test):
    train_error = []
    test_error = []
    ccp_alphas = []

    for i in range(0, 1000):
        ccp_alpha = i / 1000

        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        training_error = 1 - clf.score(X_train, y_train)
        testing_error = 1 - clf.score(X_test, y_test)
        depth = clf.get_depth()
        if depth <= 1:
            break
        train_error.append(training_error)
        test_error.append(testing_error)
        ccp_alphas.append(ccp_alpha)

    print_best_values(test_error, train_error, ccp_alphas, "CCP ALPHA", "CCP Alpha", "tree_stats")
    plot_test_and_train_error(test_error, train_error, ccp_alphas, "CCP Alpha", 1)


if __name__ == "__main__":
    main()
