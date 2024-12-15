# prober.py

import torch
import torch.nn as nn
from typing import Tuple

class Prober(nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: Tuple[int, ...]):
        """
        Args:
            embedding (int): Dimension of the input embeddings.
            arch (str): Hyphen-separated string defining layer sizes, e.g., "512-256-128".
            output_shape (tuple): Shape of the output, e.g., (2,).
        """
        super(Prober, self).__init__()
        layers = []
        layer_sizes = list(map(int, arch.split('-')))
        input_size = embedding
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers.append(nn.Linear(input_size, output_shape[-1]))  # Final output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the Prober.
        
        Args:
            x (torch.Tensor): Input embeddings [B, T, D]
        
        Returns:
            torch.Tensor: Predicted locations [B, T, 2]
        """
        B, T, D = x.shape
        x = x.view(B * T, D)
        out = self.network(x)
        out = out.view(B, T, -1)
        return out
