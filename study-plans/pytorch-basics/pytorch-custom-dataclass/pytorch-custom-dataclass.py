import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        self.data = data
        self.label_col = label_col

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        features = self.data[idx][:self.label_col] + self.data[idx][self.label_col + 1:]
        features = torch.tensor(features, dtype = torch.float32)
        label = [self.data[idx][self.label_col]]
        label = torch.tensor(label, dtype = torch.float32)
        return features, label 
