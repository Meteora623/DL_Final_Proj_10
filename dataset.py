# dataset.py

from typing import NamedTuple, Optional, List
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        probing: bool = False,
        device: str = "cuda",
        augment: bool = False,  # Added augment parameter
    ):
        self.device = device
        self.probing = probing
        self.augment = augment
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        # Define data augmentation transforms
        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.augmentation_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> WallSample:
        # Make a copy to ensure the NumPy array is writable
        states = torch.from_numpy(self.states[i].copy()).float()
        actions = torch.from_numpy(self.actions[i]).float()

        if self.augment:
            # Apply augmentation to each frame
            augmented_states = []
            for frame in states:
                frame_pil = frame.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                frame_aug = self.augmentation_transforms(frame_pil)  # [C, H, W]
                augmented_states.append(frame_aug)
            states = torch.stack(augmented_states)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i].copy()).float()
        else:
            locations = torch.empty(0).float()

        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path: str,
    probing: bool = False,
    device: str = "cuda",
    batch_size: int = 64,
    train: bool = True,
    augment: bool = False,  # Added augment parameter
) -> torch.utils.data.DataLoader:
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        augment=augment,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,  # Set to False to avoid CUDA pin_memory error
        num_workers=0,     # Set to 0 to avoid CUDA re-initialization issues
    )

    return loader
