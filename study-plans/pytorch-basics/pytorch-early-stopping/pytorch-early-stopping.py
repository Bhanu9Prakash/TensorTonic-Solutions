import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses, val_losses = [], []
    epochs_without_improvement = 0
    best_val_loss = float("inf")
    stopped_epoch =max_epochs

    for epoch in range(max_epochs):
        model.train()

        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
    
            pred = model(x)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)

        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()

        val_loss = total_loss/len(val_loader)
        val_losses.append(val_loss)

        # Early Stopping
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
                