# rl/features_simclr.py
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from models.simclr import ConvEncoder

class SimCLRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, feature_dim: int = 128):
        super().__init__(observation_space, features_dim=feature_dim)

        # Observation shape is (C, H, W)
        n_channels = observation_space.shape[0]  # ✔ must be 3

        self.encoder = ConvEncoder(input_channels=n_channels, feature_dim=feature_dim)

    def load_pretrained(self, path: str, device=None):
        if device is None:
            device = next(self.encoder.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state_dict)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, observations):
        # observations already (B, C, H, W) thanks to VecTransposeImage
        x = observations
        with torch.no_grad():
            z = self.encoder(x)
        return z