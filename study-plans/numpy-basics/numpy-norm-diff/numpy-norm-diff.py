import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    """
    Returns a float64 array of absolute normalized differences.
    """
    a = np.clip(np.array(a), lo, hi)
    b = np.clip(np.array(b), lo, hi)
    norm_diff = np.abs(a-b)/(hi - lo)
    return norm_diff
