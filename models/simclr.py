import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        # Dynamically compute conv output shape (for 56×56 inputs)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 56, 56)
            h = self.conv(dummy)
            self._conv_out = h.reshape(1, -1).size(1)

        # Fully-connected layer → final feature vector
        self.fc = nn.Linear(self._conv_out, feature_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)   # reshape is safer than view()
        z = self.fc(h)
        return z


class SimCLR(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128, proj_dim=64):
        super().__init__()

        # Shared encoder
        self.encoder = ConvEncoder(input_channels, feature_dim)

        # Projection head (SimCLR standard 2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)  # L2 normalize for contrastive loss
        return z