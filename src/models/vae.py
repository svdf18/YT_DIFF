# src/models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from src.configs.model_config import ModelConfig

class ConvBlock(nn.Module):
    """
    Basic convolutional block for the encoder and decoder.
    Consists of Conv2d -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class AudioVAE(BaseModel):
    """
    Variational Autoencoder for audio spectrograms.
    
    Architecture:
    - Encoder: Converts spectrogram to latent distribution (mu, logvar)
    - Latent: Samples from the distribution using reparameterization trick
    - Decoder: Reconstructs spectrogram from latent vector
    """
    def __init__(self, config=None):
        super().__init__()

        # Use default config if none provided
        if config is None:
            self.logger.info("Using default VAE configuration")
            config = ModelConfig.VAE_CONFIG
        
        # Extract config parameters
        self.latent_dim = config['latent_dim']
        self.hidden_dims = config['hidden_dims']
        self.in_channels = config['in_channels']
        self.n_mels = config['n_mels']
        self.kld_weight = config['kld_weight']
        
        self.logger.info(f"Initializing AudioVAE with latent dim: {self.latent_dim}")
        
        # Calculate the number of downsampling steps
        self.num_layers = len(self.hidden_dims)
        # Calculate the final encoded dimension after all downsampling
        self.encoded_dim = self.n_mels // (2 ** self.num_layers)
        
        # Build Encoder
        modules = []
        in_channels = self.in_channels
        
        # Convolutional layers
        for h_dim in self.hidden_dims:
            modules.append(
                ConvBlock(in_channels, h_dim, 
                         kernel_size=3, stride=2, padding=1)
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened dimension for FC layers
        self.flatten_dim = self._get_flat_dim()
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # Build Decoder
        modules = []
        
        # Linear projection from latent space
        self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)
        
        # Reverse the encoder architecture
        hidden_dims = self.hidden_dims[::-1]
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1,
                                     output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True))
            )
        
        # Final layer to reconstruct spectrogram
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                 kernel_size=3, stride=2, padding=1,
                                 output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dims[-1], self.in_channels,
                         kernel_size=3, padding=1),
                nn.Tanh())
        )
        
        self.decoder = nn.Sequential(*modules)
        
        # Add these lines to store spatial dimensions
        self.encoder_spatial_dims = None

        self.logger.info(f"AudioVAE initialized with {self.count_parameters():,} parameters")
    
    def _get_flat_dim(self):
        """Calculate the flattened dimension after encoder convolutions."""
        x = torch.randn(1, self.in_channels, self.n_mels, self.n_mels)
        with torch.no_grad():
            x = self.encoder(x)
            self.encoder_spatial_dims = (x.size(2), x.size(3))  # Store for decoder
        return x.flatten(1).shape[1]
    
    def encode(self, x):
        """
        Encode the input spectrogram to latent distribution parameters.
        
        Args:
            x (Tensor): Input spectrogram [batch_size, channels, n_mels, time]
            
        Returns:
            tuple(Tensor, Tensor): Mean and log-variance of latent distribution
        """
        x = self.encoder(x)
        # Store spatial dimensions before flattening
        self.encoder_spatial_dims = (x.size(2), x.size(3))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z):
        """
        Decodes latent vectors back to images.
        
        Args:
            z (Tensor): Latent vectors [B x latent_dim]
        """
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, self.encoder_spatial_dims[0], self.encoder_spatial_dims[1])
        x = self.decoder(x)
        
        # Ensure output size matches input size
        if x.size(-1) != self.n_mels:
            x = F.interpolate(x, size=(self.n_mels, self.n_mels), mode='bilinear', align_corners=False)
        
        return x
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu (Tensor): Mean of the latent Gaussian
            log_var (Tensor): Log variance of the latent Gaussian
            
        Returns:
            Tensor: Sampled latent vectors
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (Tensor): Input spectrogram [batch_size, channels, n_mels, time]
            
        Returns:
            tuple: (reconstruction, input, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, x, mu, log_var
    
    def loss_function(self, recons, x, mu, log_var, kld_weight=None):
        """
        VAE loss function = Reconstruction loss + KL divergence.
        
        Args:
            recons (Tensor): Reconstructed spectrogram
            x (Tensor): Input spectrogram
            mu (Tensor): Latent mean
            log_var (Tensor): Latent log-variance
            kld_weight (float): Weight for KL divergence term. If None, uses config value.
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kld_loss)
        """
        if kld_weight is None:
            kld_weight = self.kld_weight
            
        # Reconstruction loss (mean squared error)
        recons_loss = F.mse_loss(recons, x)
        
        # KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        
        # Total loss
        loss = recons_loss + kld_weight * kld_loss
        
        return loss, recons_loss, kld_loss
    
    def get_device(self):
        """Get the device the model is on"""
        # Get the device of the first parameter of the model
        device = next(self.parameters()).device
        # Return just the base device type without index
        return torch.device(device.type)