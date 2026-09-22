import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean = torch.mean(X, dim = 0)
    var = torch.var(X, dim = 0, correction = 0)
    denom = (var + eps)**0.5
    num = X - mean
    return gamma*(num/denom) + beta
