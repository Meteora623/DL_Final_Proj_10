# dataset.py
from typing import NamedTuple, Optional
import torch
import numpy as np

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
        print("Start loading states.npy...")
        self.states = np.load(f"{data_path}/states.npy")  # 如果非常慢可能卡在这行
        print("states.npy loaded with shape:", self.states.shape)

        print("Start loading actions.npy...")
        self.actions = np.load(f"{data_path}/actions.npy")
        print("actions.npy loaded with shape:", self.actions.shape)

        if probing:
            print("Start loading locations.npy...")
            self.locations = np.load(f"{data_path}/locations.npy")
            print("locations.npy loaded with shape:", self.locations.shape)
        else:
            self.locations = None

        self.device = device

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # 不在这里转移GPU。先在CPU上处理。
        states = torch.from_numpy(self.states[i]).float()
        actions = torch.from_numpy(self.actions[i]).float()

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float()
        else:
            locations = torch.empty(0)

        return WallSample(states=states, locations=locations, actions=actions)
