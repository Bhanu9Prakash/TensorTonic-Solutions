import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    np_data = np.array(data, dtype=np.float64)
    mask = (np_data > threshold).astype(np.float64)
    n1 = np.any(mask, axis = 1)
    n2 = np.all(mask, axis = 1)
    n1_filtered = np.where(n1[:, np.newaxis], np_data, 0.0)
    n2_filtered = np.where(n2[:, np.newaxis], np_data, 0.0)
    return np.stack([mask, n1_filtered, n2_filtered])