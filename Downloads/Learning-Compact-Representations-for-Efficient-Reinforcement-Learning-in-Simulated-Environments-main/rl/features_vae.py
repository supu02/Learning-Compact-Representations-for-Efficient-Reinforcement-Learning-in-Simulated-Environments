# rl/features_vae.py
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from models.vae import VAEEncoder


class VAEFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor pour PPO utilisant l'encodeur du VAE.
    On utilise le vecteur mu (moyenne de q(z|x)) comme représentation latente.
    """

    def __init__(self, observation_space: spaces.Box, latent_dim: int = 64):
        # features_dim = taille du vecteur renvoyé à la policy
        super().__init__(observation_space, features_dim=latent_dim)

        # Observation shape: (C, H, W), après VecTransposeImage
        shape = observation_space.shape
        assert len(shape) == 3, f"Expected (C,H,W), got {shape}"
        n_channels, H, W = shape

        if H != W:
            raise ValueError(f"Images must be square for this VAE encoder, got H={H}, W={W}")

        # On instancie seulement l'encodeur du VAE
        self.encoder = VAEEncoder(
            input_channels=n_channels,
            latent_dim=latent_dim,
            input_size=H,  # 56 dans MiniGrid-Empty-8x8-v0
        )

    def load_pretrained(self, path: str, device=None):
        """
        Charge des poids pré-entraînés (train_vae.py) et fige l'encodeur.
        """
        if device is None:
            device = next(self.encoder.parameters()).device

        state_dict = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state_dict)

        # On fige les poids pour que PPO n'update pas l'encodeur
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, C, H, W)
        On renvoie mu comme feature vector (sans sampling).
        """
        x = observations  # déjà (B, C, H, W)
        with torch.no_grad():
            mu, logvar = self.encoder(x)
        return mu
