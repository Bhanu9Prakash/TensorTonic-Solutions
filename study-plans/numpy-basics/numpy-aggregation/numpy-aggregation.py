import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    data = np.array(data)
    mean = np.mean(data, axis)
    std = np.std(data, axis)
    min = np.min(data, axis)
    max = np.max(data, axis)

    return np.stack([mean, std, min, max])
