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
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        return self.prober(e)


class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
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
        # states: [B, 1, C, H, W], actions: [B, T-1, 2]
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
    def __init__(self, model, device="cuda", lr=1e-3, momentum=0.99,
                 vicreg_lambda=0.01, vicreg_mu=0.01):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.momentum = momentum
        # VICReg超参数
        self.vicreg_lambda = vicreg_lambda  # variance penalty weight
        self.vicreg_mu = vicreg_mu          # covariance penalty weight

    def vicreg_loss(self, x):
        # x: [B,T,D]
        # Flatten batch dimension
        B, T, D = x.shape
        x = x.view(B*T, D)
        x = x - x.mean(dim=0, keepdim=True)

        # Variance term: encourage each dimension to have std > 1
        # std along batch
        eps = 1e-4
        std = torch.sqrt(x.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1 - std))

        # Covariance term: off-diagonal elements of covariance matrix should be small
        # Cov matrix: [D,D]
        cov = (x.T @ x) / (B*T - 1)
        diag = torch.diag(cov)
        cov[range(D), range(D)] = 0.0
        cov_loss = cov.pow(2).mean()

        return var_loss, cov_loss

    def train_step(self, states, actions):
        # states: [B,T,C,H,W], actions: [B,T-1,2]
        self.model.train()
        B, T, C, H, W = states.shape
        with torch.no_grad():
            target_embs = []
            for t in range(T):
                obs_t = states[:, t]
                t_enc = self.model.encode_target(obs_t)
                target_embs.append(t_enc)
            target_embs = torch.stack(target_embs, dim=1)  # [B,T,D]

        pred_encs = self.model(states=states[:,0:1], actions=actions) # [B,T,D]
        mse_loss = F.mse_loss(pred_encs, target_embs)

        # 加入VICReg正则避免表示坍缩
        var_loss, cov_loss = self.vicreg_loss(pred_encs)
        # 总loss = MSE + lambda*var_loss + mu*cov_loss
        loss = mse_loss + self.vicreg_lambda * var_loss + self.vicreg_mu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model._update_target()
        return loss.item()
