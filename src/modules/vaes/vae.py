from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from modules.formats.format import DualDiffusionFormat

class IsotropicGaussianDistribution:
    """Isotropic Gaussian distribution with diagonal covariance matrix"""

    def __init__(self, parameters: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> None:
        """Initialize distribution with parameters and log-variance
        
        Args:
            parameters: Mean of the distribution
            logvar: Log-variance of the distribution
            deterministic: If True, disable sampling and use mean only
        """
        self.deterministic = deterministic
        self.parameters = self.mean = parameters
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)
    
    def sample(self) -> torch.Tensor:
        """Sample from the distribution"""
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        """Get mode (mean) of the distribution"""
        return self.mean
    
    def kl(self, other: Optional["IsotropicGaussianDistribution"] = None) -> torch.Tensor:
        """Compute KL divergence with another distribution or standard normal
        
        Args:
            other: Optional target distribution. If None, use standard normal
        """
        if self.deterministic:
            return torch.Tensor([0.], device=self.parameters.device, dtype=self.parameters.dtype)
        
        reduction_dims = tuple(range(0, len(self.mean.shape)))
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1. - self.logvar, dim=reduction_dims)
        else:
            return 0.5 * torch.mean(
                (self.mean - other.mean).square() / other.var
                + self.var / other.var - 1. - self.logvar + other.logvar,
                dim=reduction_dims,
            )

@dataclass
class DualDiffusionVAEConfig(DualDiffusionModuleConfig, ABC):
    """Base configuration for VAE architectures"""
    in_channels: int = 2          # Number of input channels
    out_channels: int = 2         # Number of output channels
    latent_channels: int = 4      # Number of latent channels
    label_dim: int = 1           # Dimension of class labels
    dropout: float = 0.          # Dropout probability
    target_snr: float = 16.      # Target signal-to-noise ratio

class DualDiffusionVAE(DualDiffusionModule, ABC):
    """Abstract base class for VAE architectures"""
    
    module_name: str = "vae"

    @abstractmethod
    def get_class_embeddings(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Get embeddings for class conditioning"""
        pass

    @abstractmethod
    def get_recon_loss_logvar(self) -> torch.Tensor:
        """Get log variance for reconstruction loss"""
        pass
    
    @abstractmethod
    def get_target_snr(self) -> float:
        """Get target signal-to-noise ratio"""
        pass
    
    @abstractmethod
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        """Get shape of latent representation"""
        pass

    @abstractmethod
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        """Get shape of reconstructed sample"""
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> IsotropicGaussianDistribution:
        """Encode input to latent distribution"""
        pass
    
    @abstractmethod
    def decode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> torch.Tensor:
        """Decode latent representation to output"""
        pass

    def compile(self, **kwargs) -> None:
        """Compile encode and decode methods"""
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.encode = torch.compile(self.encode, **kwargs)
            self.decode = torch.compile(self.decode, **kwargs)
