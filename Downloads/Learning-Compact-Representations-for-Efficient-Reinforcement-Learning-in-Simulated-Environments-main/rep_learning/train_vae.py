# rep_learning/train_vae.py
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.vae import ConvVAE


class FramesDataset(Dataset):
    def __init__(self, frames_path: str):
        """
        Charge les frames sauvegardées par envs/collect_frames.py
        Format attendu : npz avec clé 'frames' de shape (N, H, W, C), uint8.
        """
        data = np.load(frames_path)["frames"]  # (N, H, W, C)
        self.frames = data.astype(np.float32) / 255.0  # normalisation [0,1]

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = self.frames[idx]  # (H, W, C)
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        return torch.from_numpy(img)


def train_vae(
    frames_path: str = "data/frames/frames.npz",
    latent_dim: int = 64,
    beta: float = 1.0,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 20,
    device: str | None = None,
    save_path: str = "saved_models/vae_encoder.pt",
):
    """
    Entraîne un VAE convolutionnel sur les frames MiniGrid, puis sauvegarde l'encodeur seul.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset & dataloader
    dataset = FramesDataset(frames_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Récupérer la taille d'image pour configurer le VAE
    _, H, W = dataset[0].shape  # (C, H, W)
    assert H == W, f"Images non carrées: H={H}, W={W}, adapte ConvVAE si besoin."
    input_size = H

    # Modèle
    model = ConvVAE(input_channels=3, latent_dim=latent_dim, input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training ConvVAE with latent_dim={latent_dim}, beta={beta}, input_size={input_size}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch = batch.to(device)  # (B, C, H, W)

            optimizer.zero_grad()

            x_rec, mu, logvar, z = model(batch)
            loss, recon_loss, kl_loss = ConvVAE.loss_function(
                x_rec, batch, mu, logvar, beta=beta
            )

            loss.backward()
            optimizer.step()

            bs = batch.size(0)
            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kl += kl_loss.item() * bs

        epoch_loss = total_loss / len(dataset)
        epoch_recon = total_recon / len(dataset)
        epoch_kl = total_kl / len(dataset)

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Total: {epoch_loss:.6f} | Recon: {epoch_recon:.6f} | KL: {epoch_kl:.6f}"
        )

    # Sauvegarde uniquement l'encodeur pour le RL
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved VAE encoder weights to {save_path}")


if __name__ == "__main__":
    train_vae()
