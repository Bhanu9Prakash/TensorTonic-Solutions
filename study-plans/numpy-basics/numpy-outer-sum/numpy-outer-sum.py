import numpy as np

def outer_sum(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    np_a = np.array(a)
    np_b = np.array(b)
    return np_a[:, None] + np_b[None, :]