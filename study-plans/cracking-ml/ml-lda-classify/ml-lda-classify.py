import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test = np.array(X_test)
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    means, priors = {}, {}
    centered = []
    for c in classes:
        X_c = X_train[y_train == c] # n_subset, n_features
        means[c] = np.mean(X_c, axis = 0) # n_subset, n_features
        priors[c] = len(X_c)/len(X_train)
        centered.append((X_c - means[c]))

    centered = np.vstack(centered) # n_classes*n_subset, n_features
    cov = (centered.T@centered)/(n_samples - len(classes)) + 1e-6

    predictions = []

    for x in X_test:
        scores = []
        for c in classes:
            mean = means[c]
            w = np.linalg.solve(cov, mean)
            log_prior = np.log(priors[c])
            score = (x.T@w) - (0.5*mean.T@w) + log_prior
            scores.append(score)
        prediction = classes[np.argmax(scores)]
        predictions.append(prediction)

    return predictions
        