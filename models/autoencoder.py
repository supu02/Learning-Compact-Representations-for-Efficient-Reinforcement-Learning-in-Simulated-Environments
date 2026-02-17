import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        # Dynamic conv output shape
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 56, 56)
            h = self.conv(dummy)
            self._conv_out = h.reshape(1, -1).size(1)

        self.fc = nn.Linear(self._conv_out, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, output_channels=3, latent_dim=64):
        super().__init__()

        # same conv structure for shape inference
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 56, 56)
            conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(inplace=True),
            )
            h = conv(dummy)
            C, H, W = h.shape[1:]
            self.initial_shape = (C, H, W)

        self.fc = nn.Linear(latent_dim, C * H * W)

        # decoder
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.reshape(z.size(0), *self.initial_shape)
        x_rec = self.deconv(h)
        return x_rec


class AutoEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z