from typing import NamedTuple, List, Any, Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from dataclasses import dataclass
from dataset import WallDataset
from normalizer import Normalizer
from configs import ConfigBase
from schedulers import Scheduler, LRSchedule
from models import Prober


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbingEvaluator:
    def __init__(self, device, model, probe_train_ds, probe_val_ds, config, quick_debug=False):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.config = config
        self.quick_debug = quick_debug
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model
        config = self.config
        epochs = config.epochs

        prober = Prober(repr_dim, config.prober_arch, output_shape=(2,)).to(self.device)
        optimizer = torch.optim.Adam(prober.parameters(), config.lr)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training prober"):
            for batch in tqdm(dataset, desc="Prober training step"):
                states = batch.states.to(self.device)
                actions = batch.actions.to(self.device)
                locations = batch.locations.to(self.device)

                embeddings = model(states, actions)
                pred_locations = prober(embeddings.view(-1, repr_dim))

                target_locations = locations.view(-1, 2)
                loss = criterion(pred_locations, target_locations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Loss: {loss.item()}")

        return prober

    def evaluate_all(self, prober):
        losses = {}
        for key, val_ds in self.val_ds.items():
            losses[key] = self.evaluate_pred_prober(prober, val_ds)
        return losses

    def evaluate_pred_prober(self, prober, val_ds):
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(val_ds, desc="Evaluating prober"):
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            locations = batch.locations.to(self.device)

            embeddings = self.model(states, actions)
            pred_locations = prober(embeddings.view(-1, self.model.repr_dim))
            target_locations = locations.view(-1, 2)

            loss = nn.MSELoss()(pred_locations, target_locations)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches
