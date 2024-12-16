from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        # 原 repr_dim=128 已修改为256以匹配train.py和main.py
        self.repr_dim = repr_dim
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, repr_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(True),
            nn.Linear(repr_dim, repr_dim),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256, momentum=0.99):
        super().__init__()
        self.repr_dim = repr_dim
        self.momentum = momentum

        self.online_encoder = Encoder(repr_dim=repr_dim)
        self.online_predictor = Predictor(repr_dim=repr_dim)

        self.target_encoder = Encoder(repr_dim=repr_dim)
        self._update_target(1.0)

    @torch.no_grad()
    def _update_target(self, beta=None):
        if beta is None:
            beta = self.momentum
        for tp, op in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            tp.data = beta * tp.data + (1 - beta) * op.data

    def encode_online(self, obs):
        return self.online_encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs):
        return self.target_encoder(obs)

    def forward(self, states, actions):
        B, Tm1, _ = actions.shape
        T = Tm1 + 1
        s0 = self.encode_online(states[:,0])
        preds = [s0]
        s = s0
        for t in range(T-1):
            a_t = actions[:, t]
            s = self.online_predictor(s, a_t)
            preds.append(s)
        return torch.stack(preds, dim=1)


class JEPATrainer:
    def __init__(self, model, device="cuda", lr=1e-3, momentum=0.99):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.momentum = momentum

    def train_step(self, states, actions):
        self.model.train()
        B, T, C, H, W = states.shape
        with torch.no_grad():
            target_embs = []
            for t in range(T):
                obs_t = states[:, t]
                t_enc = self.model.encode_target(obs_t)
                target_embs.append(t_enc)
            target_embs = torch.stack(target_embs, dim=1)

        pred_encs = self.model(states=states[:,0:1], actions=actions)
        loss = F.mse_loss(pred_encs, target_embs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model._update_target()
        return loss.item()
