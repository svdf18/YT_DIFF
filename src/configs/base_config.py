# src/configs/base_config.py
@dataclass
class AudioConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    n_mels: int = 256
    hop_length: int = 512
    
@dataclass
class TrainingConfig:
    batch_size: int = 8  # Smaller for MPS
    learning_rate: float = 1e-4
    num_epochs: int = 100