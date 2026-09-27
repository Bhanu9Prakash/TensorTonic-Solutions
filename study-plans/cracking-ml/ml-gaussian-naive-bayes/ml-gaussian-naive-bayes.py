import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)

    mean, variance, prior = {}, {}, {}

    for c in classes:
        X_c = X_train[y_train == c]

        mean[c] = np.mean(X_c, axis=0)
        variance[c] = np.var(X_c, axis=0) + 1e-9
        prior[c] = len(X_c) / len(X_train)

    predictions = []

    for example in X_test:
        score = []

        for c in classes:
            mean_c = mean[c]
            variance_c = variance[c]

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance_c)
                + ((example - mean_c) ** 2) / variance_c
            )

            log_prior = np.log(prior[c])

            score.append(log_prior + log_likelihood)

        prediction = classes[np.argmax(score)]
        predictions.append(prediction)

    return predictions