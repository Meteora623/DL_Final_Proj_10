from typing import NamedTuple, List, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from normalizer import Normalizer
from models import Prober
from configs import ConfigBase
from schedulers import Scheduler, LRSchedule

@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"

def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse

class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = ProbingConfig(),
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        dataset = self.ds
        model = self.model
        config = self.config
        epochs = config.epochs
        if self.quick_debug:
            epochs = 1

        test_batch = next(iter(dataset))
        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            model.repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = list(prober.parameters())
        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)
        scheduler = Scheduler(
            schedule=config.schedule,
            base_lr=config.lr,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
        )
        total_steps = epochs * len(dataset)
        scheduler.set_total_steps(total_steps)
        step = 0

        for epoch in tqdm(range(epochs), desc="Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                init_states = batch.states[:, 0:1].to(self.device, non_blocking=True)
                actions = batch.actions.to(self.device, non_blocking=True)
                target = batch.locations.to(self.device, non_blocking=True)

                pred_encs = model(states=init_states, actions=actions)
                pred_encs = pred_encs.transpose(0,1)

                pred_encs = pred_encs.detach()

                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                target = self.normalizer.normalize_location(target)

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )
                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 2, device=self.device)

                    indices_all = torch.randperm(n_steps, device=self.device)[:config.sample_timesteps]
                    for i in range(bs):
                        sampled_pred_encs[:, i, :] = pred_encs[indices_all, i, :]
                        sampled_target_locs[i, :] = target[i, indices_all]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                optimizer_pred_prober.zero_grad()
                per_probe_loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)
                step += 1

                if self.quick_debug and step > 2:
                    break
            if self.quick_debug:
                break

        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober=prober, val_ds=val_ds, prefix=prefix)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        prober.eval()
        model = self.model
        probing_losses = []
        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            init_states = batch.states[:, 0:1].to(self.device, non_blocking=True)
            actions = batch.actions.to(self.device, non_blocking=True)
            target = batch.locations.to(self.device, non_blocking=True)

            pred_encs = model(states=init_states, actions=actions)
            pred_encs = pred_encs.transpose(0, 1)

            target = self.normalizer.normalize_location(target)

            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)
        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss
