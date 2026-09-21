import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    """
    Returns float64 correlation matrices for a, b, and their combined rows.
    """
    a = np.array(a)
    b = np.array(b)
    corr_a = np.corrcoef(a, rowvar = False)
    corr_b = np.corrcoef(b, rowvar = False)
    corr_c = np.corrcoef(np.vstack([a, b]), rowvar = False)
    
    return np.stack([corr_a, corr_b, corr_c]) 
