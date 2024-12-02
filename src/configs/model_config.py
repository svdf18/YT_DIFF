from dataclasses import dataclass, field
from .base_config import AudioConfig
from .base_config import TrainingConfig

@dataclass
class VAEConfig(TrainingConfig):
    """Configuration specific to the VAE model"""
    pass 

@dataclass
class ModelConfig:
    """Configuration for all model components"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)