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
    kld_weight: float = 0.000001

@dataclass
class EDM2Config:
    enabled: bool = True
    # Architecture
    model_type: Literal["edm2_unet"] = "edm2_unet"
    attention_type: Literal["sdp", "einsum"] = "sdp"
    use_torch_compile: bool = True
    hidden_dims: list = field(default_factory=lambda: [32, 64, 128, 256])
    
    # EDM2 specific
    rank_reduction_factor: float = 0.25
    condition_modulation: bool = True
    dropout_rate: float = 0.1
    dropout_preserve_magnitude: bool = True
    time_embedding_dim: int = 128  # Add this line for time embedding dimension
    
    # Noise Schedule
    noise_schedule: Literal["edm", "ddim"] = "edm"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    
    # Training specific
    use_ema: bool = True
    ema_lengths: list = field(default_factory=lambda: [0.9999, 0.9995, 0.999])
    stratified_sampling: bool = True
    classifier_free_guidance: bool = True
    guidance_scale: float = 3.0
    label_dropout_prob: float = 0.1

@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "src/tests/test_data/processed"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    save_interval: int = 10
    gradient_accumulation_steps: int = 4
    lr_warmup_steps: int = 5000
    lr_decay_exponent: float = 1.0
    
    # Window parameters
    window_length: int = 80
    hop_length: int = 40

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    edm2: EDM2Config = field(default_factory=EDM2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)