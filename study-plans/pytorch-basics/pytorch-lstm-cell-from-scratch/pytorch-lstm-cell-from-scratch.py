import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ii = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.b_ii = nn.Parameter(torch.randn((hidden_size,)))
        self.W_hi = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_hi = nn.Parameter(torch.randn((hidden_size,)))

        self.W_if = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.b_if = nn.Parameter(torch.randn((hidden_size,)))
        self.W_hf = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_hf = nn.Parameter(torch.randn((hidden_size,)))

        self.W_ig = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.b_ig = nn.Parameter(torch.randn((hidden_size,)))
        self.W_hg = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_hg = nn.Parameter(torch.randn((hidden_size,)))

        self.W_io = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.b_io = nn.Parameter(torch.randn((hidden_size,)))
        self.W_ho = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_ho = nn.Parameter(torch.randn((hidden_size,)))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hidden state, cell state), both float32 tensors of shape (batch, hidden_size).
        """
        i_t = torch.sigmoid(((x@self.W_ii.T) + self.b_ii) + ((h_prev@self.W_hi.T) + self.b_hi)) # b, h
        f_t = torch.sigmoid(((x@self.W_if.T) + self.b_if) + ((h_prev@self.W_hf.T) + self.b_hf)) # b, h
        g_t = torch.tanh(((x@self.W_ig.T) + self.b_ig) + ((h_prev@self.W_hg.T) + self.b_hg))   # b, h
        o_t = torch.sigmoid(((x@self.W_io.T) + self.b_io) + ((h_prev@self.W_ho.T) + self.b_ho)) #b, h

        c_t = f_t*c_prev + i_t*g_t
        h_t = o_t*torch.tanh(c_t)
            
        return h_t, c_t
