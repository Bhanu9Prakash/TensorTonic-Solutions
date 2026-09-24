import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.b_ih = nn.Parameter(torch.randn((hidden_size)))
        self.W_hh = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_hh = nn.Parameter(torch.randn((hidden_size)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 hidden-state tensor of shape (batch, hidden_size).
        """
        a = x@self.W_ih.T + self.b_ih
        b = h_prev@self.W_hh.T + self.b_hh

        return torch.tanh(a + b)
