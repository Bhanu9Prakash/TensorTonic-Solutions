import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    """
    Returns a dictionary: weights (float list), bias (float), predictions (integer list).
    """
    X, y = np.array(X), np.array(y)
    n_examples, n_features = X.shape
    w = np.zeros((n_features))
    b = 0.0
    
    wagg, bagg = np.zeros((n_features)), 0.0
    count = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            score = xi@w + b
            if yi*score <= 0:
                w += lr*yi*xi
                b += lr*yi
            wagg += w
            bagg += b
            count += 1

    wavg, bavg = wagg/count, bagg/count
    scores = X@wavg + bavg
    predictions = np.where(scores > 0, 1, -1)

    wavg = np.round(wavg, 4)
    bavg = np.round(bavg, 4)

    result = {"weights": wavg.tolist(), "bias": float(bavg), "predictions": predictions.astype(int).tolist()}
            
    return result

