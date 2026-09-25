import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model//num_heads
        self.W_q = nn.Parameter(torch.randn((d_model, d_model)))
        self.W_k = nn.Parameter(torch.randn((d_model, d_model)))
        self.W_v = nn.Parameter(torch.randn((d_model, d_model)))
        self.W_o = nn.Parameter(torch.randn((d_model, d_model)))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = Q.shape
        Q_proj = Q@self.W_q
        K_proj = K@self.W_k
        V_proj = V@self.W_v

        Q = Q_proj.view(batch, seq_len, self.h, self.d_k).transpose(1, 2) # B, H, S, D
        K = K_proj.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        V = V_proj.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)

        scores = Q@K.transpose(-2, -1)
        scores /= self.d_k**0.5
        scores = torch.softmax(scores, dim = -1)
        attention = scores@V
        attention = attention.transpose(1, 2)
        attention = attention.contiguous().view(batch, seq_len, d_model)
        y = attention@self.W_o
        
        return y
        
