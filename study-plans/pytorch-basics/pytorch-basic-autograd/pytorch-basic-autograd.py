import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    values = torch.tensor(values, requires_grad = True)
    y = torch.sum(values**3 + 2*values)
    y.backward()
    return values.grad
