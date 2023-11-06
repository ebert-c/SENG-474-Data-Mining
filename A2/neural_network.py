from sklearn.neural_network import MLPClassifier
import A2.utils.data_loader as dl
from A2.utils.k_fold_cross_validation import k_fold
import A2.utils.graphing as graph
import matplotlib.pyplot as plt

def main():
    X_train, y_train, X_test, y_test = dl.load_data()
    vary_number_hidden_layers(X_train, y_train, X_test, y_test)
    #vary_activation(X_train, y_train, X_test, y_test)
    #vary_hidden_layer_size(X_train, y_train, X_test, y_test)


def determine_optimal_hyperparameters(X_train, y_train):
    activations = ['identity', 'logistic', 'tanh', 'relu']
    risks = []
    configs = []
    for layer_size in range(1, 201, 25):
        for hidden_layer_num in range(1, 3):
            for activation in activations:
                layers = [layer_size] * hidden_layer_num
                nn = MLPClassifier(activation=activation, hidden_layer_sizes=layers)
                nn.fit(X_train, y_train)
                risks.append(k_fold(nn, X_train, y_train))
                configs.append((layer_size, hidden_layer_num, activation))
    min_risk_index = risks.index(min(risks))
    print(configs[min_risk_index])


def vary_hidden_layer_size(X_train, y_train, X_test, y_test):
    risks = []
    sizes = []
    for i in range(1, 201, 25):
        nn = MLPClassifier(hidden_layer_sizes=(i,), activation='identity')
        nn.fit(X_train, y_train)
        risks.append(k_fold(nn, X_test, y_test))
        sizes.append(i)
    graph.plot_risk(risks, sizes, "Hidden Layer Sizes(NN)")
    min_risk_index = risks.index(min(risks))
    print(sizes[min_risk_index])


def vary_number_hidden_layers(X_train, y_train, X_test, y_test):
    risks = []
    sizes = []
    for i in range(1, 10):
        layers = [100] * i
        nn = MLPClassifier(hidden_layer_sizes=layers, activation='identity')
        nn.fit(X_train, y_train)
        risks.append(k_fold(nn, X_test, y_test))
        sizes.append(i)
    graph.plot_risk(risks, sizes, "Hidden Layer Number(NN)")
    min_risk_index = risks.index(min(risks))
    print(sizes[min_risk_index])


def vary_activation(X_train, y_train, X_test, y_test):
    risks = []
    sizes = []
    activations = ['identity', 'logistic', 'tanh', 'relu']
    for i in activations:
        nn = MLPClassifier(activation=i)
        nn.fit(X_train, y_train)
        risks.append(k_fold(nn, X_test, y_test))
        sizes.append(i)

    plt.bar(activations, risks)
    plt.ylabel("Error")
    plt.savefig("Nonlinearity (NN).svg", format="svg")
    min_risk_index = risks.index(min(risks))
    print(sizes[min_risk_index])


if __name__ == "__main__":
    main()
