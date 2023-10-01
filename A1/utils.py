import matplotlib.pyplot as plt


def plot_test_and_train_error(test_error, train_error, x_axis, xlabel, i):
    plt.figure(i)
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(f"{xlabel}.svg", format="svg")


def print_best_values(test_error, train_error, variants, var_name, title, filename):
    min_test_error = min(test_error)
    best_test_size = variants[test_error.index(min_test_error)]

    min_train_error = min(train_error)
    best_train_size = variants[train_error.index(min_train_error)]

    with open(f"{filename}.txt", "a") as f:
        f.write(title + "\n")
        f.write(f"Minimum Test Error: {min_test_error} achieved when {var_name} = {best_test_size}\n")
        f.write(f"Minimum Training Error: {min_train_error} achieved when {var_name} = {best_train_size}\n")
