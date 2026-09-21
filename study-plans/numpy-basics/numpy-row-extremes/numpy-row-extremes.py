import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    data = np.array(data, dtype = np.float64)
    maxima = np.max(data, axis = 1)
    max_ind = np.argmax(data, axis = 1)
    minima = np.min(data, axis = 1)
    min_ind = np.argmin(data, axis = 1)
    return np.stack([maxima, max_ind, minima, min_ind])
