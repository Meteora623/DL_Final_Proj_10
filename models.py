from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim
from vit_encoder import ViTEncoder

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        return self.prober(e)

class Predictor(nn.Module):
    def __init__(self, repr_dim=64, action_dim=2):
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
    def __init__(self, repr_dim=64, momentum=0.99):
        super().__init__()
        self.repr_dim = repr_dim
        self.momentum = momentum

        self.online_encoder = ViTEncoder(image_size=65, patch_size=13, dim=repr_dim, depth=2, heads=2, mlp_ratio=4)
        self.online_predictor = Predictor(repr_dim=repr_dim)

        self.target_encoder = ViTEncoder(image_size=65, patch_size=13, dim=repr_dim, depth=2, heads=2, mlp_ratio=4)
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
            s = s / (s.norm(dim=-1, keepdim=True) + 1e-6)
            preds.append(s)
        return torch.stack(preds, dim=1)

class JEPATrainer:
    def __init__(self, model, device="cuda", lr=1e-4, momentum=0.99,
                 vicreg_lambda=0.1, vicreg_mu=0.1):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.momentum = momentum
        self.vicreg_lambda = vicreg_lambda
        self.vicreg_mu = vicreg_mu

    def vicreg_loss(self, x):
        B, T, D = x.shape
        x = x.view(B*T, D)
        x = x - x.mean(dim=0, keepdim=True)
        eps = 1e-4
        std = torch.sqrt(x.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1 - std))
        cov = (x.T @ x) / (B*T - 1)
        cov[range(D), range(D)] = 0.0
        cov_loss = cov.pow(2).mean()
        return var_loss, cov_loss

    def train_step(self, states, actions):
        self.model.train()
        B, T, C, H, W = states.shape
        with torch.no_grad():
            target_embs = [self.model.encode_target(states[:, t]) for t in range(T)]
            target_embs = torch.stack(target_embs, dim=1)

        pred_encs = self.model(states=states[:,0:1], actions=actions)
        mse_loss = F.mse_loss(pred_encs, target_embs)

        var_loss, cov_loss = self.vicreg_loss(pred_encs)
        loss = mse_loss + self.vicreg_lambda * var_loss + self.vicreg_mu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model._update_target()
        return loss.item()
