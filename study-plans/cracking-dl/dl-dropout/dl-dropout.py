import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """
    if mode == "test":
        return X
        
    X, mask = np.array(X), np.array(mask)
    p = drop_prob


    return (X*mask)/(1 - p)
