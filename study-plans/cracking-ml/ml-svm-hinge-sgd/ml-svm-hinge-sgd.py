import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Returns fitted parameters and training predictions.
    """
    X, y = np.array(X), np.array(y)
    n_samples, n_features = X.shape
    w = np.zeros((n_features))
    b = 0.0
    for _ in range(n_epochs):
        for xi, yi in zip(X, y):
            score = xi@w + b
            margin = yi*score
            if margin < 1:
                w -= lr*(lam*w - yi*xi)
                b += lr*yi
            else:
                w -= lr*lam*w
                
    scores = X@w + b
    predictions = np.where(scores > 0, 1, -1)
    w = np.round(w, 4)
    b = np.round(b, 4)
    result = {"weights": w.tolist(), "bias": b.item(), "predictions": predictions.astype(int).tolist()}
    return result