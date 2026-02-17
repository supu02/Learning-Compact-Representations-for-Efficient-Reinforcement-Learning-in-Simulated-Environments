import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from models.simclr import SimCLR

############ STABLE, SAFE, CORRECTED INFO-NCE LOSS ############

def info_nce_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = z @ z.T                                   # (2B, 2B)
    mask = torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
    sim = sim / temperature

    # pour chaque i, le positif est (i + B) % (2B)
    labels = (torch.arange(2*batch_size, device=z.device) + batch_size) % (2*batch_size)

    # on masque les self-similarities
    sim = sim.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss



###############################################################


class SimCLRDataset(Dataset):
    def __init__(self, frames_path):
        data = np.load(frames_path)["frames"]  # (N,H,W,C)
        self.frames = data.astype(np.float32) / 255.0

        H = self.frames.shape[1]

        # SimCLR-like augmentations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(size=H, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = (self.frames[idx] * 255).astype(np.uint8)

        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2


def train_simclr(
    frames_path="data/frames/frames.npz",
    feature_dim=128,
    proj_dim=64,
    batch_size=64,
    lr=1e-3,
    num_epochs=20,
    device=None,
    save_path="saved_models/simclr_encoder.pt",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SimCLRDataset(frames_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SimCLR(input_channels=3, feature_dim=feature_dim, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x1, x2 in tqdm(dataloader, desc=f"SimCLR Epoch {epoch+1}/{num_epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = info_nce_loss(z1, z2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x1.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

    # Save only encoder (not projection head)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved SimCLR encoder to {save_path}")


if __name__ == "__main__":
    train_simclr()