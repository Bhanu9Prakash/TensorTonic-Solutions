import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    losses = []
    lrs = []
    
    for _ in range(num_epochs):
        train_loss = 0.0
        for x, t in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, t)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)
        losses.append(train_loss)

        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
        

    res = {"losses": losses, 
          "lrs": lrs}
    return res
