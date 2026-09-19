import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    np_data = np.array(data, dtype=np.float64)[row_idx]
    np_clipped = np.clip(np_data, lo, hi)
    
    return np.vstack((np_data, np_clipped))