# normalizer.py
import torch
import numpy as np


class Normalizer:
    def __init__(self, data_loader=None):
        if data_loader is not None:
            all_locations = []
            for batch in data_loader:
                if batch.locations.numel() > 0:
                    all_locations.append(batch.locations)
            if all_locations:
                all_locations = torch.cat(all_locations, dim=0)  # [N, T, 2]
                self.mean = all_locations.mean(dim=(0, 1), keepdim=True)
                self.std = all_locations.std(dim=(0, 1), keepdim=True) + 1e-6  # Prevent division by zero
                print(f"Computed Normalizer Mean: {self.mean}")
                print(f"Computed Normalizer Std: {self.std}")
            else:
                self.mean = torch.zeros(1, 1, 2)
                self.std = torch.ones(1, 1, 2)
                print("No locations found in data_loader. Using default mean and std.")
        else:
            # Use predefined mean and std
            self.mean = torch.tensor([31.5863, 32.0618]).view(1, 1, 2)
            self.std = torch.tensor([16.1025, 16.1353]).view(1, 1, 2)
            print(f"Using Predefined Normalizer Mean: {self.mean}")
            print(f"Using Predefined Normalizer Std: {self.std}")

    def normalize_location(self, loc: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the location tensor.

        Args:
            loc (torch.Tensor): Tensor of shape [B, T, 2].

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return (loc - self.mean.to(loc.device)) / self.std.to(loc.device)

    def unnormalize_location(self, loc: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the location tensor.

        Args:
            loc (torch.Tensor): Normalized tensor of shape [B, T, 2].

        Returns:
            torch.Tensor: Unnormalized tensor.
        """
        return loc * self.std.to(loc.device) + self.mean.to(loc.device)

    def unnormalize_mse(self, mse):
        """
        Unnormalizes the MSE loss.

        Args:
            mse (torch.Tensor): MSE loss tensor.

        Returns:
            float: Unnormalized MSE loss.
        """
        return mse * (self.std.to(mse.device) ** 2).sum().item()
