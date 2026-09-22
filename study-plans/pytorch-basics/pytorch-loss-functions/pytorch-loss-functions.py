import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    loss = None
    if method == "mse":
        loss = torch.mean((pred - target)**2)
    elif method == "cross_entropy":
        shifted = pred - pred.max(dim=1, keepdim=True).values
        log_partition = shifted.exp().sum(dim=1).log()
        target_logits = shifted[torch.arange(pred.shape[0]), target]
        loss = (log_partition - target_logits).mean()
    elif method == "huber":
        a = torch.abs(pred - target)
        loss = torch.where(a <= delta, 0.5*a**2, delta*(a - 0.5*delta))
        loss = torch.mean(loss)

    return loss.item()
