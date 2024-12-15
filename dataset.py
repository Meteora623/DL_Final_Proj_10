# dataset.py
from typing import NamedTuple, Optional
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
        data_path,
        probing=False,
        device="cuda",
        augment=False,  # Added augment parameter
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

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float()
        actions = torch.from_numpy(self.actions[i]).float()

        if self.augment:
            # Apply augmentation to each frame
            augmented_states = []
            for frame in states:
                frame_pil = frame.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                frame_aug = self.augmentation_transforms(frame_pil)  # [C, H, W]
                augmented_states.append(frame_aug)
            states = torch.stack(augmented_states)

        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
    augment=False,  # Added augment parameter
):
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
        pin_memory=True,  # Changed to True for better performance on GPU
        num_workers=4,  # Adjust based on your system
    )

    return loader
