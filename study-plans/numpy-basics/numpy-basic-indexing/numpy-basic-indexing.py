import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    return np.array(arr, dtype = np.float64)[row_start:row_stop, col_start:col_stop]
