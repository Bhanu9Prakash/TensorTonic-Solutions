import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    np_d = np.array(data)
    np_w = np.array(weights)

    return np_d*np_w[:, None]