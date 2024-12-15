# normalizer.py

import torch

class Normalizer:
    def __init__(self, mean: float = None, std: float = None):
        """
        Initializes the Normalizer with optional mean and std.
        If not provided, they will be computed from the data.
        """
        self.mean = mean
        self.std = std

    def normalize_location(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Normalizes location data.

        Args:
            locations (torch.Tensor): [B, T, 2]

        Returns:
            torch.Tensor: Normalized locations
        """
        if self.mean is None or self.std is None:
            # Compute mean and std from the data
            self.mean = locations.mean()
            self.std = locations.std()
        return (locations - self.mean) / self.std

    def unnormalize_mse(self, mse: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the MSE loss.

        Args:
            mse (torch.Tensor): Normalized MSE loss

        Returns:
            torch.Tensor: Unnormalized MSE loss
        """
        return mse * (self.std ** 2)
