import numpy as np

def tile_diff(data: list, reps: int) -> np.ndarray:
    """
    Returns float64 slices of tiled values and next-row differences with a final zero row.
    """
    data = np.array(data)
    tiled = np.tile(data, (reps, 1))
    diff = np.diff(tiled, axis = 0)
    diff = np.vstack([diff, np.zeros((1, data.shape[1]))])
    
    return np.stack([tiled, diff])
    
