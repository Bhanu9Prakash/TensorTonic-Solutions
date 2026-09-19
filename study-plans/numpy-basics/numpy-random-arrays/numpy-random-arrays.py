import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, shape).astype(np.float64) if kind == 'normal' else rng.uniform(0, 1, shape).astype(np.float64)
