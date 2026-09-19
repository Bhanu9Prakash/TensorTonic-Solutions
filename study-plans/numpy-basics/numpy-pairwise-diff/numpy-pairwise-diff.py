import numpy as np

def pairwise_diff(a):
    """Returns: np.ndarray of shape (n, n) where out[i,j] = a[i] - a[j]"""
    np_a = np.array(a)
    return np_a[:, None] - np_a[None, :]