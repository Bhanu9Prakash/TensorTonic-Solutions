import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = {
            "lr": lr,
            "momentum": momentum
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            params = group["params"]

            for param in params:
                
                if param.grad is None:
                    continue
                    
                grad = param.grad
                state = self.state[param]

                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(param)

                velocity = state["velocity"]

                velocity *= momentum
                velocity += grad

                param -= lr*velocity


        return loss
