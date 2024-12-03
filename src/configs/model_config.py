from dataclasses import dataclass, field
from typing import Optional
from .base_config import AudioConfig, VAEConfig, EDM2Config, TrainingConfig, Config

@dataclass
class VAEOnlyConfig(Config):
    """Configuration with only VAE enabled"""
    def __post_init__(self):
        self.edm2 = EDM2Config(enabled=False)

@dataclass
class EDM2OnlyConfig(Config):
    """Configuration with only EDM2 enabled"""
    def __post_init__(self):
        self.vae = VAEConfig(enabled=False)

@dataclass
class HybridConfig(Config):
    """Configuration with both VAE and EDM2 enabled"""
    pass  # Uses default settings from Config

# You can also create a factory method if you prefer
def create_config(model_type: str) -> Config:
    """Factory method to create appropriate config based on model type"""
    configs = {
        "vae": VAEOnlyConfig,
        "edm2": EDM2OnlyConfig,
        "hybrid": HybridConfig,
    }
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(configs.keys())}")
    return configs[model_type]()

@dataclass
class ModelConfig:
    """Configuration for all model components"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)