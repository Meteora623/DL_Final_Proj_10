# models.py

import torch
import torch.nn as nn

class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super(JEPAModel, self).__init__()
        # Example Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Assuming RGB images
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, repr_dim),  # Adjust based on input image size
            nn.ReLU()
        )
        # Example Action Processing
        self.action_processor = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, repr_dim),
            nn.ReLU()
        )
        # Example Future Prediction (Dummy, replace with actual implementation)
        self.future_predictor = nn.Linear(repr_dim, repr_dim)
    
    def forward(self, states, actions):
        """
        Args:
            states (torch.Tensor): [B, T, C, H, W]
            actions (torch.Tensor): [B, T, A]
        
        Returns:
            torch.Tensor: [B, T, D]
        """
        B, T, C, H, W = states.shape
        states = states.view(B * T, C, H, W)  # Merge B and T for processing
        actions = actions.view(B * T, -1)
        
        state_encodings = self.encoder(states)  # [B*T, D]
        action_encodings = self.action_processor(actions)  # [B*T, D]
        
        combined = state_encodings + action_encodings  # [B*T, D]
        future = self.future_predictor(combined)  # [B*T, D]
        
        future = future.view(B, T, -1)  # [B, T, D]
        return future
