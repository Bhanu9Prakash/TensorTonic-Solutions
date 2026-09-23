import torch

def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    if method == "xavier_uniform":
        bound = (6 / (fan_in + fan_out))**0.5
        return torch.empty((fan_out, fan_in)).uniform_(-bound, bound)
    elif method == "xavier_normal":
        std = (2/(fan_in + fan_out))**0.5
        return torch.empty((fan_out, fan_in)).normal_(mean = 0, std = std)
    elif method == "he_uniform":
        bound = (6/fan_in)**0.5
        return torch.empty((fan_out, fan_in)).uniform_(-bound, bound)
    elif method == "he_normal":
        std = (2/fan_in)**0.5
        return torch.empty((fan_out, fan_in)).normal_(mean = 0, std = std)

    return 