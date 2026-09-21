import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    data = np.array(data, dtype = np.float64)
    pad = lambda x : np.pad(x, ((pad_width, pad_width), (pad_width, pad_width)))
    round = pad(np.round(data, decimals))
    floor = pad(np.floor(data))
    ceil = pad(np.ceil(data))
    
    return np.stack([round, floor, ceil])
