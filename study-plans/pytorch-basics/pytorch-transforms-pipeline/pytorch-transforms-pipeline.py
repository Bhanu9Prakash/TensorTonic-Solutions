import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        image = torch.tensor(image, dtype = torch.float32)
        image = image/255.0
        norm_image = (image - self.mean)/self.std
        norm_image = norm_image.permute(2, 0, 1)
        return norm_image
