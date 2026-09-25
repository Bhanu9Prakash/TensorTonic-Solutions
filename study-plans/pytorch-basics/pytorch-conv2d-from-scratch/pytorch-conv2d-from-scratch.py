import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.randn((out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (N, out_channels, H-k+1, W-k+1).
        """
        N, Cin, H, W = x.shape
        K = self.kernel_size
        Cout = self.out_channels
        Hout, Wout = H - K + 1, W - K + 1

        patches = F.unfold(x, kernel_size = K) # N, Cin*K*K, Hout*Wout
        weight = self.weight.reshape(Cout, Cin*K*K) # Cout, Cin*K*K

        output = weight@patches
        output = output + self.bias.view(1, Cout, 1)
        
        output = output.reshape((N, Cout, Hout, Wout))
        
        return output
