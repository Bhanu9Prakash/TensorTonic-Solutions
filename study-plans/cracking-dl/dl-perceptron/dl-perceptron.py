import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X = np.array(X)
    y = np.array(y)
    n_features = X.shape[1]
    W = np.zeros(n_features)
    b = 0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = xi@W + b
            yi_pred = 1 if z >= 0 else 0
            error = yi - yi_pred
            W = W + lr*error*xi
            b = b + lr*error
        
    W = np.round(W, 2)
    b = np.round(b, 2)

    return W.tolist(), b
