from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from A1.utils import plot_risks, print_best_risks


def main():
    data = np.loadtxt("spambase.data", delimiter=",")
    X = [x[:-1] for x in data]
    y = [y[-1] for y in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    rfc_risk = []
    ada_risk = []
    estimators = []

    for i in range(50, 550, 50):
        rfc = RandomForestClassifier(n_estimators=i)
        ada = AdaBoostClassifier(n_estimators=i, estimator=DecisionTreeClassifier(max_depth=1))
        rfc_risk.append(k_fold(rfc, X_train, y_train))
        ada_risk.append(k_fold(ada, X_train, y_train))
        estimators.append(i)

    plot_risks(rfc_risk, "Random Forest", ada_risk, "AdaBoost", estimators, "(k-fold)Estimators", 10)
    print_best_risks(rfc_risk, ada_risk, estimators)

    min_rfc_risk = min(rfc_risk)
    min_ada_risk = min(ada_risk)
    best_rfc_estimator = estimators[rfc_risk.index(min_rfc_risk)]
    best_ada_estimator = estimators[ada_risk.index(min_ada_risk)]

    tuned_rfc = RandomForestClassifier(n_estimators=best_rfc_estimator)
    tuned_ada = AdaBoostClassifier(n_estimators=best_ada_estimator, estimator=DecisionTreeClassifier(max_depth=1))

    tuned_rfc.fit(X_train, y_train)
    tuned_rfc_error = 1 - tuned_rfc.score(X_test, y_test)
    tuned_ada.fit(X_train, y_train)
    tuned_ada_error = 1 - tuned_ada.score(X_test, y_test)

    plt.figure(11)
    plt.bar(["Random Forest", "AdaBoost"], [tuned_rfc_error, tuned_ada_error])
    plt.ylabel("Risk")
    plt.savefig("(k-fold)Tuned Error Comparison.svg", format="svg")


def k_fold(algorithm, X_train, y_train):
    K = 8
    x_folds = []
    y_folds = []
    fold_len = int(len(X_train) / K)
    errors = []

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
        errors.append(1 - algorithm.score(x_folds[k], y_folds[k]))
    risk = sum(errors) / K
    return risk


if __name__ == "__main__":
    main()
