# src/configs/base_config.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    n_mels: int = 80
    hop_length: int = 512
    in_channels: int = 1
    
@dataclass
class VAEConfig:
    enabled: bool = True
    latent_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [32, 64, 128, 256])
    in_channels: int = 1
    kld_weight: float = 0.01

@dataclass
class EDM2Config:
    enabled: bool = True
    # Architecture
    model_type: Literal["edm2_unet"] = "edm2_unet"
    attention_type: Literal["sdp", "einsum"] = "sdp"
    use_torch_compile: bool = True
    hidden_dims: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    
    # ResNet specific (new)
    resnet_groups: int = 8
    resnet_time_scale: float = 1.0
    use_pixel_norm: bool = True
    mlp_dim_mult: int = 2
    resnet_dropout: float = 0.1
    
    # EDM2 specific
    rank_reduction_factor: float = 0.5
    condition_modulation: bool = True
    dropout_rate: float = 0.2
    dropout_preserve_magnitude: bool = True
    time_embedding_dim: int = 256
    
    # Noise Schedule
    noise_schedule: Literal["edm", "ddim"] = "edm"
    sigma_min: float = 0.001
    sigma_max: float = 120.0
    rho: float = 9.0
    
    # Training specific
    use_ema: bool = True
    ema_lengths: list = field(default_factory=lambda: [0.9999, 0.9995, 0.999])
    stratified_sampling: bool = True
    classifier_free_guidance: bool = True
    guidance_scale: float = 5.0
    label_dropout_prob: float = 0.2

@dataclass
class TrainingConfig:
    # Data
    train_dir: str = "data/training/processed"
    val_dir: str = "data/validation/processed"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 100
    save_interval: int = 10
    gradient_accumulation_steps: int = 16
    lr_warmup_steps: int = 10000
    lr_decay_exponent: float = 1.0
    
    # Window parameters
    window_length: int = 80
    hop_length: int = 40
    
    # Validation
    early_stopping_patience: int = 50
    validation_interval: int = 5
    
    # Visualization
    vis_interval: int = 5  # Visualize every N epochs
    vis_samples: int = 4   # Number of samples to visualize

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    edm2: EDM2Config = field(default_factory=EDM2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)