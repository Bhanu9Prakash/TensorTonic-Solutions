import numpy as np

def winsorize(data: list, lo_q: float, hi_q: float) -> np.ndarray:
    """
    Returns float64 slices of clipped values, lower mask, and upper mask.
    """
    data = np.array(data)
    lower = np.percentile(data, lo_q, axis = 0, method = 'linear')
    upper = np.percentile(data, hi_q, axis = 0, method = 'linear')
    clipped = np.clip(data, lower, upper)    
    lower_b = data < lower
    upper_b = data > upper
    
    return np.stack([clipped, lower_b, upper_b])
