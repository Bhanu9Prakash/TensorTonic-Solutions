import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 attention tensor of shape (batch, seq_q, d_v).
    """
    scores = Q@K.transpose(-2, -1)
    d_k = Q.shape[-1]
    scores /= d_k**0.5
    attention = torch.softmax(scores, dim = -1)
    
    return attention@V 
