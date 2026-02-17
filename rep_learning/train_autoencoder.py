# rep_learning/train_autoencoder.py
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.autoencoder import AutoEncoder


class FramesDataset(Dataset):
    def __init__(self, frames_path: str):
        data = np.load(frames_path)["frames"]  # (N, H, W, C) uint8
        self.frames = data.astype(np.float32) / 255.0  # normalize to [0,1]

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = self.frames[idx]  # H,W,C
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        return torch.from_numpy(img)


def train_autoencoder(
    frames_path="data/frames/frames.npz",
    latent_dim=64,
    batch_size=128,
    lr=1e-3,
    num_epochs=20,
    device=None,
    save_path="saved_models/ae_encoder.pt"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FramesDataset(frames_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = AutoEncoder(input_channels=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - MSE loss: {epoch_loss:.6f}")

    # Save only encoder for RL
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder weights to {save_path}")


if __name__ == "__main__":
    train_autoencoder()