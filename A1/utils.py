import matplotlib.pyplot as plt


def plot_test_and_train_error(test_error, train_error, x_axis, xlabel, i):
    plt.figure(i)
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(f"{xlabel}.svg", format="svg")


def plot_risks(risk1, risk1_label, risk2, risk2_label, x_axis, xlabel, i):
    plt.figure(i)
    plt.plot(x_axis, risk1, label=risk1_label)
    plt.plot(x_axis, risk2, label=risk2_label)
    plt.xlabel(xlabel)
    plt.ylabel("Risk")
    plt.legend()
    plt.savefig(f"{xlabel}(k_cross).svg", format="svg")


def print_best_values(test_error, train_error, variants, var_name, title, filename):
    min_test_error = min(test_error)
    best_test_size = variants[test_error.index(min_test_error)]

    min_train_error = min(train_error)
    best_train_size = variants[train_error.index(min_train_error)]

    with open(f"{filename}.txt", "a") as f:
        f.write(title + "\n")
        f.write(f"Minimum Test Error: {min_test_error} achieved when {var_name} = {best_test_size}\n")
        f.write(f"Minimum Training Error: {min_train_error} achieved when {var_name} = {best_train_size}\n")


def print_best_risks(risk1, risk2, variants):
    min_risk1 = min(risk1)
    min_risk2 = min(risk2)
    best_var_1 = variants[risk1.index(min_risk1)]
    best_var_2 = variants[risk2.index(min_risk2)]
    with open("risk_stats.txt", "a") as f:
        f.write("Risk Values\n")
        f.write(f"Minimum Random Forest risk: {min_risk1} achieved when Number of Estimators = {best_var_1}\n")
        f.write(f"Minimum AdaBoost risk: {min_risk2} achieved when Number of Estimators = {best_var_2}\n")