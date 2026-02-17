import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from models.autoencoder import Encoder


class AEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, latent_dim: int = 64):
        # features_dim is the size of the latent vector
        super().__init__(observation_space, features_dim=latent_dim)

        # obs shape after VecTransposeImage is (C, H, W)
        shape = observation_space.shape  # e.g. (3, 56, 56)

        # channels = first dimension
        n_channels = shape[0]

        self.encoder = Encoder(input_channels=n_channels, latent_dim=latent_dim)

    def load_pretrained(self, path: str, device=None):
        if device is None:
            device = next(self.encoder.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state_dict)
        # freeze weights
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations come as (B, C, H, W) after VecTransposeImage
        x = observations  # already (B, C, H, W)
        with torch.no_grad():
            z = self.encoder(x)
        return z