import torch
import torch.nn as nn

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

        output = torch.zeros((N, Cout, Hout, Wout))

        for n in range(N):
            for out in range(Cout):
                for h in range(Hout):
                    for w in range(Wout):
                        param = x[n, :, h:h+K, w:w+K]
                        kernel = self.weight[out]
                        bias = self.bias[out]

                        output[n, out, h, w] = (param*kernel).sum() + bias
                        
        
        return output
