import matplotlib.pyplot as plt


def plot_test_and_train_error(test_error, train_error, x_axis, x_label):
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(x_label)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(f"{x_label}.svg", format="svg")


def plot_risk(risk, x_axis, x_label):
    plt.scatter(x_axis, risk, label="Error")
    plt.xlabel(x_label)
    plt.ylabel("Risk")
    plt.legend()
    plt.savefig(f"{x_label}.svg", format="svg")
