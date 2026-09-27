import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
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
        dw = (2/n_examples)*(X.T@error) + alpha*np.sign(weights)
        db = (2/n_examples)*np.sum(error)
        weights -= lr*dw
        bias -= lr*db
        
    
    return [round(value, 4) for value in weights], round(bias, 4)
