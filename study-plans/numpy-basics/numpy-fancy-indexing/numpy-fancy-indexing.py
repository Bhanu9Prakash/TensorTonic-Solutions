import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    np_arr = np.array(arr, dtype = np.float64)
    
    return np_arr[indices, :] if axis == 0 else np_arr[:, indices]