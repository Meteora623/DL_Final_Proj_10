# evaluator.py

from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Prober  # Ensure Prober is correctly imported
from normalizer import Normalizer
from configs import ProbingConfig
import matplotlib.pyplot as plt
import os


class ProbingEvaluator:
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        probe_train_ds: DataLoader,
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

        self.normalizer = Normalizer()  # Initialized without arguments

    def train_pred_prober(self) -> Prober:
        """
        Probes whether the predicted embeddings capture the future locations
        """
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        try:
            test_batch = next(iter(dataset))
            prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        except StopIteration:
            raise ValueError("The probe_train_ds is empty.")

        prober = Prober(
            embedding=repr_dim,
            arch=config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size if hasattr(dataset, 'batch_size') else 64
        batch_steps = None

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################################
                # Forward pass through your JEPA model
                init_states = batch.states[:, 0:1]  # [B, 1, C, H, W]
                pred_encs = model(states=init_states, actions=batch.actions)  # [B, T, D]
                pred_encs = pred_encs.transpose(0, 1)  # [T, B, D]

                # Ensure pred_encs has shape [T, B, D]
                ################################################################################

                pred_encs = pred_encs.detach()

                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                losses_list = []

                target = getattr(batch, "locations").cuda()
                target = self.normalizer.normalize_location(target)

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    # we only randomly sample n timesteps to train prober.
                    # we most likely do this to avoid OOM
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )

                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 2).to(self.device)

                    for i in range(bs):
                        indices = torch.randperm(n_steps)[: config.sample_timesteps]
                        sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                        sampled_target_locs[i, :] = target[i, indices]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs

                # TODO: Handle forward pass here
                # --------------------------------------------------------------------
                # Modified Forward Pass to align pred_locs with target
                # --------------------------------------------------------------------
                prober.train()  # Ensure prober is in training mode
                pred_locs = prober(pred_encs)  # [B, T, 2]

                # --------------------------------------------------------------------
                # Ensure pred_locs and target have the same shape
                # --------------------------------------------------------------------
                assert pred_locs.shape == target.shape, f"Shape mismatch after prober: pred_locs {pred_locs.shape}, target {target.shape}"

                # --------------------------------------------------------------------
                # Debugging: Print shapes to verify
                # --------------------------------------------------------------------
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step+1}:")
                    print(f"pred_locs shape: {pred_locs.shape}")
                    print(f"target shape: {target.shape}")

                # Compute loss
                losses = self.location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"normalized pred locations loss {per_probe_loss.item()}")

                losses_list.append(per_probe_loss)
                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(
        self,
        prober: Prober,
    ) -> dict:
        """
        Evaluates on all the different validation datasets
        """
        avg_losses = {}

        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=prefix,
            )

        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober: Prober, val_ds: DataLoader, prefix: str = "") -> float:
        """
        Evaluates the prober on a single validation dataset.
        """
        probing_losses = []
        prober.eval()  # Ensure prober is in evaluation mode

        # Create a directory to save sample plots
        sample_dir = f"sample_predictions_{prefix}"
        os.makedirs(sample_dir, exist_ok=True)

        for idx, batch in enumerate(tqdm(val_ds, desc=f"Evaluating on {prefix}")):
            ################################################################################
            # Forward pass through your JEPA model
            init_states = batch.states[:, 0:1]  # [B, 1, C, H, W]
            pred_encs = self.model(states=init_states, actions=batch.actions)  # [B, T, D]
            pred_encs = pred_encs.transpose(0, 1)  # [T, B, D]
            ################################################################################

            target = getattr(batch, "locations").cuda()
            target = self.normalizer.normalize_location(target)

            # TODO: Handle forward pass here
            # --------------------------------------------------------------------
            # Modified Forward Pass to align pred_locs with target
            # --------------------------------------------------------------------
            prober.eval()  # Ensure prober is in evaluation mode
            pred_locs = prober(pred_encs)  # [B, T, 2]

            # --------------------------------------------------------------------
            # Ensure pred_locs and target have the same shape
            # --------------------------------------------------------------------
            assert pred_locs.shape == target.shape, f"Shape mismatch after prober: pred_locs {pred_locs.shape}, target {target.shape}"

            # --------------------------------------------------------------------
            # Debugging: Print shapes to verify
            # --------------------------------------------------------------------
            print(f"Evaluation - {prefix} - Batch {idx+1}:")
            print(f"pred_locs shape: {pred_locs.shape}")
            print(f"target shape: {target.shape}")

            # Compute loss
            losses = self.location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

            # --------------------------------------------------------------------
            # Visualize a few samples
            # --------------------------------------------------------------------
            if idx == 0:  # Visualize only the first batch
                for i in range(min(5, pred_locs.shape[0])):  # Visualize up to 5 samples
                    plt.figure(figsize=(6, 6))
                    plt.scatter(target[i, :, 0].cpu(), target[i, :, 1].cpu(), label='Target', c='blue')
                    plt.scatter(pred_locs[i, :, 0].cpu(), pred_locs[i, :, 1].cpu(), label='Predicted', c='red')
                    plt.title(f"Sample {i+1} - Batch {idx+1}")
                    plt.legend()
                    plt.xlabel("X Coordinate")
                    plt.ylabel("Y Coordinate")
                    plt.savefig(os.path.join(sample_dir, f"sample_{i+1}_batch_{idx+1}.png"))
                    plt.close()

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)

        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss

    @staticmethod
    def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        mse = (pred - target).pow(2).mean()
        return mse
