import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    data = TensorDataset(features, labels)

    class_counts = torch.bincount(labels)
    class_weights = 1.0/class_counts.float()
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples = len(data), replacement = True)

    dataloader = DataLoader(data, batch_size = batch_size, sampler = sampler)

    return dataloader
