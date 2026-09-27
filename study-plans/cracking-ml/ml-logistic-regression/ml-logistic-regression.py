import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X)
    y = np.array(y)
    n_examples, n_features = X.shape
    weights = np.zeros((n_features))
    bias = 0.0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for _ in range(n_iters):
        z = X@weights + bias
        z = np.clip(z, -500, 500)
        y_pred = sigmoid(z)
        error = y_pred - y
        dw = (1/n_examples)*(X.T@error)
        db = (1/n_examples)*(np.sum(error))
        weights -= lr*dw
        bias -= lr*db
        

    return [round(value, 4) for value in weights], round(bias, 4)
