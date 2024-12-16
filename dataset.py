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
        self.device = device
        # 移除 mmap_mode
        self.states = np.load(f"{data_path}/states.npy")  # 原：np.load(..., mmap_mode='r')
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # 这里直接返回 CPU Tensor，然后在 collate_fn 或之后再迁移到GPU，减少阻塞
        states = torch.from_numpy(self.states[i]).float()
        actions = torch.from_numpy(self.actions[i]).float()

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float()
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

    # 使用num_workers加快数据加载, 开启pin_memory
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=True,
        num_workers=4,  # 可根据实际情况调整
    )

    return loader
