import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    np_d = np.array(data)
    np_w = np.array(weights)[None, :]
    return np_d*np_w
    