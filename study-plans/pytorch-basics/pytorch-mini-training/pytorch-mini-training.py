import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    if not model.training:
        model.train()
    average_loss = []
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        average_loss.append(loss)
        loss.backward()
        optimizer.step()
        
    return torch.tensor(average_loss).mean().item()
