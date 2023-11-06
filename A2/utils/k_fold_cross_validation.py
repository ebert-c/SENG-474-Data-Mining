def k_fold(algorithm, X_train, y_train):
    K = 5
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