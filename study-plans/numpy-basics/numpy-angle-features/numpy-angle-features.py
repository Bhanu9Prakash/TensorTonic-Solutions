import numpy as np

def angle_features(angles: list) -> np.ndarray:
    """
    Returns a (3, n) float64 array with sine, cosine, and tangent rows.
    """
    angles = np.array(angles)
    sin = np.sin(angles)
    cos = np.cos(angles)
    tan = np.tan(angles)
    return np.stack([sin, cos, tan])
