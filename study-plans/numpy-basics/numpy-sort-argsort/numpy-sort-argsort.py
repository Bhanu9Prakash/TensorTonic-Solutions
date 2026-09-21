import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:
    """
    Returns a (2, m, n) float64 array of sorted values and source indices.
    """
    data = np.array(data, dtype = np.float64)
    sort = np.sort(data, axis)
    argsort = np.argsort(data, axis)

    return np.stack([sort, argsort])
