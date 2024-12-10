from typing import Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from modules.formats.format import DualDiffusionFormat

@dataclass
class DualDiffusionUNetConfig(DualDiffusionModuleConfig, ABC):
    """Base configuration for UNet architectures"""
    in_channels: int = 4
    out_channels: int = 4
    use_t_ranges: bool = False
    inpainting: bool = False
    label_dim: int = 1
    dropout: float = 0.
    sigma_max: float = 200.
    sigma_min: float = 0.03
    sigma_data: float = 1.

class DualDiffusionUNet(DualDiffusionModule, ABC):
    """Abstract base class for UNet architectures"""
    
    module_name: str = "unet"

    @abstractmethod
    def get_class_embeddings(self, class_labels: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        """Get embeddings for class conditioning"""
        pass
    
    @abstractmethod
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None,
            class_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get log variance for sigma loss"""
        pass
    
    @abstractmethod
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        """Get shape of latent representation"""
        pass

    @abstractmethod
    @torch.no_grad()
    def convert_to_inpainting(self) -> None:
        """Convert model for inpainting"""
        pass

    @abstractmethod
    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: DualDiffusionFormat,
                class_embeddings: Optional[torch.Tensor] = None,
                t_ranges: Optional[torch.Tensor] = None,
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        pass