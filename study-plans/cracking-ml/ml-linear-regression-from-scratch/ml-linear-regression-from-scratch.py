import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X)
    y = np.array(y)
    n_examples, n_features = X.shape
    weights = np.zeros((n_features))
    bias = 0.0
    for _ in range(epochs):
        y_pred = X@weights + bias
        error = y_pred - y
        dw = (2/n_examples)*(X.T@error)
        db = (2/n_examples)*np.sum(error)
        weights -= lr*dw
        bias -= lr*db

    return [round(value, 4) for value in weights], round(bias, 4)
