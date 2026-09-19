import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    np_data = np.array(data)
    mean = np.mean(np_data, axis = 0)
    std = np.std(np_data, axis = 0)

    return (np_data - mean[None, :])/std[None, :]