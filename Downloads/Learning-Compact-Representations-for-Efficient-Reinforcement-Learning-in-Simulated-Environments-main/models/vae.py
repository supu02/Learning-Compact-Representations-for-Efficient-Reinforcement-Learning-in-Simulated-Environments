# models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 64, input_size: int = 56):
        """
        CNN encoder for a convolutional VAE.
        Sort deux vecteurs : mu et logvar (pour la distribution q(z|x)).
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),              # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # 14 -> 7
            nn.ReLU(inplace=True),
        )

        # Calcul dynamique de la taille en sortie de la partie conv
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            h = self.conv(dummy)
            self._conv_shape = h.shape[1:]           # (C, H, W)
            #self._conv_out = h.view(1, -1).size(1)   # C * H * W
            self._conv_out = h.reshape(1, -1).size(1)

        # Deux têtes linéaires : mu et logvar
        self.fc_mu = nn.Linear(self._conv_out, latent_dim)
        self.fc_logvar = nn.Linear(self._conv_out, latent_dim)

    def forward(self, x):
        """
        x : (B, C, H, W) dans [0,1]
        Retourne mu, logvar
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @property
    def conv_shape(self):
        return self._conv_shape

    @property
    def conv_out_dim(self):
        return self._conv_out


class VAEDecoder(nn.Module):
    def __init__(self, output_channels: int = 3, latent_dim: int = 64, conv_shape=None):
        """
        CNN decoder qui reconstruit une image à partir d'un latent z.
        conv_shape : (C, H, W) attendu avant les ConvTranspose2d.
        """
        super().__init__()

        if conv_shape is None:
            # Valeur par défaut compatible avec l'encoder ci-dessus pour input_size=56
            conv_shape = (128, 7, 7)

        self._conv_shape = conv_shape
        C, H, W = conv_shape
        self.fc = nn.Linear(latent_dim, C * H * W)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 14 -> 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.Sigmoid(),  # sortie dans [0,1]
        )

    def forward(self, z):
        """
        z : (B, latent_dim)
        Retourne x_rec : (B, C, H, W)
        """
        h = self.fc(z)
        h = h.view(z.size(0), *self._conv_shape)
        x_rec = self.deconv(h)
        return x_rec


class ConvVAE(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 64, input_size: int = 56):
        """
        VAE convolutionnel complet : Encoder + Decoder.
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            input_size=input_size,
        )
        self.decoder = VAEDecoder(
            output_channels=input_channels,
            latent_dim=latent_dim,
            conv_shape=self.encoder.conv_shape,
        )

    def reparameterize(self, mu, logvar):
        """
        Trick de reparamétrisation : z = mu + sigma * epsilon.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Retourne :
        - x_rec : reconstruction
        - mu, logvar : paramètres de q(z|x)
        - z : latent échantillonné
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar, z

    @staticmethod
    def loss_function(x_rec, x, mu, logvar, beta: float = 1.0):
        """
        Loss VAE classique : reconstruction + beta * KL.
        x, x_rec : (B, C, H, W) dans [0,1]
        """
        # Reconstruction MSE (tu peux aussi tester BCE si tu veux)
        recon_loss = F.mse_loss(x_rec, x, reduction="mean")

        # KL divergence entre q(z|x) et N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.sum(dim=1).mean()  # moyenne sur le batch

        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss
