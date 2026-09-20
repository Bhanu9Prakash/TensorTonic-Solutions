import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X = np.array(X)
    y = np.array(y)
    W = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        for xi, yi_true in zip(X, y):
            z = xi.T@W + b
            yi_pred = 1 if z >= 0 else 0
            error = yi_true - yi_pred
            W = W + lr*error*xi
            b = b + lr*error

    return W.tolist(), b
