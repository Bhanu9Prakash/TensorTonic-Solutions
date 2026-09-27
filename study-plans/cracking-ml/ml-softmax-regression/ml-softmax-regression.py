import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.array(X)
    y = np.array(y)
    n_examples, n_features = X.shape
    weights = np.zeros((n_features, num_classes))
    bias = np.zeros(num_classes)

    y_one_hot = np.eye(num_classes)[y]

    for _ in range(n_iters):
        logits = X@weights + bias # n_examples, num_classes
        logits -= np.max(logits, axis = 1, keepdims = True)
        exp_logits = np.exp(logits)
        y_pred = exp_logits/np.sum(exp_logits, axis = 1, keepdims = True)
        error = y_pred - y_one_hot # n_examples, num_classes
        dw = (1/n_examples)*(X.T@error) # n_features, num_classes
        db = (1/n_examples)*np.sum(error, axis = 0) # num_classes

        weights -= lr*dw
        bias -= lr*db

    weights = np.round(weights, 4)
    bias = np.round(bias, 4)

    return weights.tolist(), bias.tolist()
