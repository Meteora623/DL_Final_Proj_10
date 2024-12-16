import torch
import numpy as np
from typing import NamedTuple
from torchvision import transforms
from torch.utils.data import DataLoader

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(self, data_path, probing=False, device="cuda"):
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

        # 数据增强：随机裁剪和水平翻转
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=65, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, i):
        states = self.states[i]  # np array [T, C, H, W]
        # 转换为tensor后对每帧进行增广
        # states shape: (T, 2, 65, 65)
        states_tensor = torch.empty_like(torch.from_numpy(states), dtype=torch.float)
        for t in range(states.shape[0]):
            frame = torch.from_numpy(states[t]).float() # [2, 65, 65]
            # 转换到PIL后需要C,H,W->H,W,C
            frame_img = transforms.functional.to_pil_image(frame)
            frame_aug = self.transform(frame_img) # [C,H,W]
            states_tensor[t] = frame_aug

        actions = torch.from_numpy(self.actions[i]).float()
        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float()
        else:
            locations = torch.empty(0)

        return WallSample(states=states_tensor, locations=locations, actions=actions)

def create_wall_dataloader(data_path, probing=False, device="cuda", batch_size=64, train=True):
    ds = WallDataset(data_path=data_path, probing=probing, device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=True, pin_memory=True, num_workers=4)
    return loader
