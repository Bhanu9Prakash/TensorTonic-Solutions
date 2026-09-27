import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)
    means, variances, priors = {}, {}, {}
    
    for c in classes:
        X_c = X_train[y_train == c]
        means[c], variances[c] = np.mean(X_c, axis = 0), np.var(X_c, axis = 0) + 1e-9
        priors[c] = len(X_c)/len(X_train)

    predictions = []

    for x in X_test:
        scores = []
        for c in classes:
            mean, variance = means[c], variances[c]
            log_likelihood = -0.5*((np.log(2*np.pi*variance)) + ((x - mean)**2/(variance)) )
            log_likelihood = np.sum(log_likelihood)
            log_prior = np.log(priors[c])
            scores.append(log_likelihood + log_prior)
        prediction = classes[np.argmax(scores)]
        predictions.append(prediction)

    return predictions
