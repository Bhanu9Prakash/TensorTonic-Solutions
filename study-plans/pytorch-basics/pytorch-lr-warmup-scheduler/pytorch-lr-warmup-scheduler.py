import math

def warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int) -> list[float]:
    """
    Returns one Python float learning rate per step.
    """
    lr = []

    for t in range(total_steps):
        lr_t = base_lr
        if t < warmup_steps:
            lr_t *= ((t + 1)/warmup_steps)     
        else:
            lr_t *= (1 + math.cos(math.pi*((t - warmup_steps)/(total_steps - warmup_steps))))/2
            
        lr.append(lr_t)
    
    return lr 
