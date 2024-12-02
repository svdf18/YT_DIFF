class ModelConfig:
    """Configuration for all model components"""
    
    # Audio processing
    SAMPLE_RATE = 44100
    N_FFT = 2048
    N_MELS = 80
    HOP_LENGTH = 512
    
    # Spectrogram Configuration
    SPEC_CONFIG = {
        'sample_rate': SAMPLE_RATE,
        'n_fft': N_FFT,
        'n_mels': N_MELS,
        'hop_length': HOP_LENGTH
    }
    
    # VAE Configuration
    VAE_CONFIG = {
        'latent_dim': 512,
        'hidden_dims': [32, 64, 128, 256, 512],
        'in_channels': 1,
        'n_mels': N_MELS,
        'kld_weight': 0.01
    }
    
    # Training Configuration
    TRAINING_CONFIG = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'ema_decay': 0.995
    }