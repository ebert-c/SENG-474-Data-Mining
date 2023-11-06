import A2.utils.data_loader as dl
from sklearn.linear_model import SGDClassifier
import A2.utils.graphing as graph
from sklearn.svm import SVC
from A2.utils.k_fold_cross_validation import k_fold
import matplotlib.pyplot as plt


def main():
    X_train, y_train, X_test, y_test = dl.load_data()

    linear_svm(X_train, y_train, X_test, y_test)
    gaussian_svm(X_train, y_train, X_test, y_test)


def linear_svm(X_train, y_train, X_test, y_test):
    C = 0.001
    beta = 1.5
    test_errors = []
    train_errors = []
    c = []
    for i in range(0, 21):
        C_value = C * (beta ** i)
        alpha = 1 / (len(X_train) * C_value)
        svm = SGDClassifier(alpha=alpha)
        svm.fit(X_train, y_train)
        train_errors.append(1 - svm.score(X_train, y_train))
        test_errors.append(1 - svm.score(X_test, y_test))
        c.append(C_value)
    min_risk_index = test_errors.index(min(test_errors))
    print(c[min_risk_index])
    graph.plot_test_and_train_error(test_errors, train_errors, c, "C Value (LinearSVM)")


def gaussian_svm(X_train, y_train, X_test, y_test):
    C = 1
    beta = 1.5
    G = 0.01
    g_beta = 1.5
    low_params = []

    for i in range(11):
        c_val = C * (beta ** i)
        risks = []
        gammas = []
        for j in range(11):
            gamma = G * (g_beta ** j)
            svm = SVC(C=c_val, gamma=gamma)
            risks.append(k_fold(svm, X_train, y_train))
            gammas.append(gamma)
        lowest_risk_index = risks.index(min(risks))
        low_params.append((c_val, gammas[lowest_risk_index]))

    tuned_risks = []
    for i in low_params:
        svm = SVC(C=i[0], gamma=i[1])
        tuned_risks.append(k_fold(svm, X_train, y_train))
    tuned_risk_index = tuned_risks.index(min(tuned_risks))
    optimal_gamma = low_params[tuned_risk_index]
    print(optimal_gamma)

    train_errors = []
    test_errors = []
    gammas = []

    for i in low_params:
        svm = SVC(C=i[0], gamma=i[1])
        svm.fit(X_train, y_train)
        train_errors.append(1 - svm.score(X_train, y_train))
        test_errors.append(1 - svm.score(X_test, y_test))
        gammas.append(i[1])
    plt.scatter(gammas, test_errors, label="Test Error")
    plt.scatter(gammas, train_errors, label="Training Error")
    plt.legend()
    plt.xlabel("Gamma Values")
    plt.ylabel("Error")
    plt.savefig("Gamma Value (Gaussian Kernel)")


if __name__ == "__main__":
    main()
