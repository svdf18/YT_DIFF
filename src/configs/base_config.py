# src/configs/base_config.py
from dataclasses import dataclass, field

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    n_mels: int = 80
    hop_length: int = 512
    
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
    
    # Model
    latent_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [32, 64, 128, 256])
    in_channels: int = 1
    n_mels: int = 80
    kld_weight: float = 0.000001
    
    # Window parameters
    window_length: int = 80
    hop_length: int = 40