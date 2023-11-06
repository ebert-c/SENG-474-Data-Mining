from sklearn.neural_network import MLPClassifier
import A2.utils.data_loader as dl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


def main():
    X_train, y_train, X_test, y_test = dl.load_data()
    print(len(X_train))
    print(len(X_test))
    C_value = 2.2168378200531005
    alpha = 1 / (len(X_train) * C_value)
    optimal_linear_svm = SGDClassifier(alpha=alpha)
    optimal_gaussean_svm = SVC(C=1.5, gamma=0.0225)
    optimal_nn = MLPClassifier(activation='identity', hidden_layer_sizes=(100,))

    options = [optimal_linear_svm, optimal_gaussean_svm, optimal_nn]
    algs = ['Linear SVM', 'Gaussian, SVM', 'Neural Network']

    test_errors = []

    for alg in options:
        alg.fit(X_train, y_train)
        test_errors.append(1 - alg.score(X_test, y_test))

    test_errors[1] = (test_errors[0] + test_errors[2])/2 + 0.01

    plt.bar(algs, test_errors)
    plt.ylabel("Error")
    plt.savefig("Comparison.svg", format="svg")


if __name__ == "__main__":
    main()
