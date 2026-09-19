import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """

    return np.arange(start, stop, param, dtype=np.float64) if kind == 'arange' else np.linspace(start, stop, param, dtype=np.float64)
