# Test the VAE trainer
import os
# Set MPS fallback before any other imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
from accelerate import Accelerator

from src.modules.vaes.vae_edm2 import DualDiffusionVAE_EDM2, DualDiffusionVAE_EDM2Config
from src.modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from src.training.module_trainers.vae_trainer import VAETrainer, VAETrainerConfig

if __name__ == "__main__":
    # Initialize components
    vae_config = DualDiffusionVAE_EDM2Config(
        in_channels=2,  # Stereo audio
        out_channels=2,
        latent_channels=4,
        model_channels=96,
        channel_mult=(1, 2, 3, 5),
        num_layers_per_block=3,
        dropout=0.0   # Start without dropout
    )
    
    # Create spectrogram format config
    spectrogram_config = SpectrogramFormatConfig(
        sample_rate=32000,
        step_size_ms=16,
        window_duration_ms=64,
        padded_duration_ms=64,
        num_frequencies=256,
        min_frequency=20,
        max_frequency=16000,
        freq_scale_type="mel"
    )
    
    # Initialize trainer config
    trainer_config = VAETrainerConfig(
        block_overlap=8,
        block_widths=(8, 16, 32, 64),
        channel_kl_loss_weight=0.1,
        recon_loss_weight=0.1
    )

    # Create model and format processor
    model = DualDiffusionVAE_EDM2(vae_config)
    format_processor = SpectrogramFormat(spectrogram_config)
    accelerator = Accelerator(mixed_precision="no")

    # Create trainer
    trainer = VAETrainer(
        config=trainer_config,
        model=model,
        format_processor=format_processor,
        accelerator=accelerator
    )

    # Create dummy batch for testing
    dummy_batch = {
        'audio': torch.randn(16, 2, 256, 256)  # Match training batch_size of 16
    }

    # Test training step
    outputs = trainer.training_step(dummy_batch)
    print("Training outputs:", outputs)

    # Test sample generation
    samples = trainer.generate_samples(dummy_batch)
    print("\nGenerated samples:", samples)

    # Check SNR and loss values
    print("\nTarget SNR:", trainer.model.get_target_snr())
    print("Loss values:", {k: v.item() for k, v in outputs.items() if isinstance(v, torch.Tensor)})
