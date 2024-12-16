import torch
import numpy as np
from typing import NamedTuple
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

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
        # 注：不再包含ToPILImage，因为我们手动将frame转为PIL
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=65, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, i):
        # states: [T, C, H, W]的只读mmap数组
        states = self.states[i] 
        # 创建可写副本
        states_copy = states.copy()  
        states_tensor = torch.from_numpy(states_copy).float()  # [T, C, H, W]

        # 对每帧进行数据增强
        # 先将每帧转为PIL图，再通过transform增强
        for t in range(states_tensor.shape[0]):
            frame = states_tensor[t]  # [C, H, W]
            frame_img = F.to_pil_image(frame)  # 转为PIL图像
            frame_aug = self.transform(frame_img)  # 数据增强
            states_tensor[t] = frame_aug

        # 同理对 actions 和 locations 执行 copy 
        actions_copy = self.actions[i].copy()
        actions = torch.from_numpy(actions_copy).float()

        if self.locations is not None:
            locations_copy = self.locations[i].copy()
            locations = torch.from_numpy(locations_copy).float()
        else:
            locations = torch.empty(0)

        return WallSample(states=states_tensor, locations=locations, actions=actions)

def create_wall_dataloader(data_path, probing=False, device="cuda", batch_size=64, train=True):
    ds = WallDataset(data_path=data_path, probing=probing, device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=True, pin_memory=True, num_workers=4)
    return loader
