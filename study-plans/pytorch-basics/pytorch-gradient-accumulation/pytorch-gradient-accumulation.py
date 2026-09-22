import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.detach().clone().requires_grad_(True)
    last_mean_gradient = torch.zeros_like(w)

    for i, (x, y) in enumerate(micro_batches):
        preds = torch.dot(w, x)
        loss = (preds - y)**2
        loss.backward()

        if (i + 1)%accum_steps == 0:
            mean_grad = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr*mean_grad
            last_mean_gradient = mean_grad
            w.grad.zero_()


    return w.detach(), last_mean_gradient.detach()
