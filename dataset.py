import torch
import numpy as np
from typing import NamedTuple

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        print("Start loading states.npy with mmap_mode='r'...")
        self.states = np.load(f"{data_path}/states.npy", mmap_mode='r')
        print("states.npy loaded with shape:", self.states.shape)

        print("Start loading actions.npy with mmap_mode='r'...")
        self.actions = np.load(f"{data_path}/actions.npy", mmap_mode='r')
        print("actions.npy loaded with shape:", self.actions.shape)

        if probing:
            print("Start loading locations.npy with mmap_mode='r'...")
            self.locations = np.load(f"{data_path}/locations.npy", mmap_mode='r')
            print("locations.npy loaded with shape:", self.locations.shape)
        else:
            self.locations = None

        self.device = device

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, i):
        states = torch.from_numpy(np.array(self.states[i])).float()
        actions = torch.from_numpy(np.array(self.actions[i])).float()
        if self.locations is not None:
            locations = torch.from_numpy(np.array(self.locations[i])).float()
        else:
            locations = torch.empty(0)
        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )

    return loader
