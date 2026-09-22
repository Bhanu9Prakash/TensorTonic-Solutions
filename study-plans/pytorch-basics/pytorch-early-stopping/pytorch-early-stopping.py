import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    stopped_epoch = max_epochs
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()    

        train_loss = 0.0
        for x, t in train_loader:
            optimizer.zero_grad()
            
            y = model(x)
            loss = criterion(y, t)
    
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, t in val_loader:
                y = model(x)
                loss = criterion(y, t)
                val_loss += loss.item()

        val_loss = val_loss/len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_epoch = epoch + 1
            break


    res = {"train_losses": train_losses,
          "val_losses": val_losses,
          "stopped_epoch": stopped_epoch}

    return res
