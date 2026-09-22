import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    logits = logits - torch.max(logits, dim = 1, keepdim = True).values
    exp = torch.exp(logits).to(torch.float32)
    return exp/exp.sum(dim = 1, keepdim = True)
