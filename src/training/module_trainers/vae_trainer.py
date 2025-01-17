"""
This file implements the VAE trainer class for the dual diffusion model.

The training process for the VAE follows these key steps:
1. Initialization:
   - Sets up VAE model configuration and parameters
   - Initializes loss functions and weights
   - Configures spectral loss with block sizes and overlap
   - Prepares target SNR and noise levels
   - Sets up model compilation if enabled

2. Input Processing:
   - Takes batches of audio data as input
   - Handles data formatting and device placement
   - Applies any necessary preprocessing
   - Manages batch dimensions and shapes

3. Forward Pass:
   - Encodes input data into latent space
   - Applies variational sampling in latent space
   - Decodes latents back to audio domain
   - Handles real and imaginary components

4. Loss Computation:
   - Calculates reconstruction loss in time domain
   - Computes KL divergence on channel dimensions
   - Applies spectral losses across multiple scales
   - Weights different loss components:
     * Channel KL loss (weight=0.1)
     * Imaginary component loss (weight=0.1) 
     * Point-wise loss (weight=0)
     * Reconstruction loss (weight=0.1)

5. Optimization:
   - Accumulates gradients from loss components
   - Updates VAE model parameters
   - Applies learning rate scheduling
   - Handles gradient clipping if configured

6. Validation:
   - Evaluates reconstruction quality
   - Computes validation metrics
   - Generates sample reconstructions
   - Tracks best performing checkpoints

7. Monitoring:
   - Logs training metrics and loss values
   - Records reconstruction quality metrics
   - Exports validation samples
   - Tracks training progress

"""

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Optional, Tuple

from modules.vaes.vae_edm2 import DualDiffusionVAE_EDM2
from modules.formats.spectrogram import SpectrogramFormat
from training.loss import MultiscaleSpectralLoss, MultiscaleSpectralLossConfig

@dataclass
class VAETrainerConfig:
    """Configuration for VAE training"""
    # Spectral loss settings
    block_overlap: int = 8
    block_widths: list[int] = (8, 16, 32, 64)
    mel_bands: Optional[int] = None
    freq_range: Tuple[float, float] = (20, 20000)
    sample_rate: int = 32000
    stereo_weight: float = 0.5
    
    # Loss weights
    channel_kl_loss_weight: float = 0.1
    recon_loss_weight: float = 0.1

class VAETrainer:
    def __init__(self, 
                 config: VAETrainerConfig,
                 model: DualDiffusionVAE_EDM2,
                 format_processor: SpectrogramFormat,
                 accelerator: Accelerator):
        self.config = config
        self.model = model
        self.format_processor = format_processor
        self.accelerator = accelerator

        # Initialize spectral loss with full config
        spectral_loss_config = MultiscaleSpectralLossConfig(
            block_widths=config.block_widths,
            block_overlap=config.block_overlap,
            mel_bands=config.mel_bands,
            freq_range=config.freq_range,
            sample_rate=config.sample_rate,
            stereo_weight=config.stereo_weight
        )
        self.spectral_loss = MultiscaleSpectralLoss(spectral_loss_config)

    def training_step(self, batch):
        """Single training step"""
        spectrograms = batch['audio']
        
        # VAE forward pass
        z = self.model.encode(spectrograms)
        recon = self.model.decode(z.sample())
        
        # Compute spectral loss (now handles stereo properly)
        recon_loss = self.spectral_loss(recon, spectrograms)
        kl_loss = z.kl() * self.config.channel_kl_loss_weight
        
        # Total loss
        loss = recon_loss * self.config.recon_loss_weight + kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def validation_step(self, batch):
        """Single validation step"""
        with torch.no_grad():
            metrics = self.training_step(batch)
        return metrics

    @torch.no_grad()
    def generate_samples(self, batch):
        """Generate sample reconstructions for visualization"""
        spectrograms = batch['audio']
        
        # Get reconstructions in spectrogram form
        z = self.model.encode(spectrograms)
        recon_spectrograms = self.model.decode(z.sample())
        
        # Convert spectrograms back to audio waveforms using correct method name
        original_audio = self.format_processor.sample_to_raw(spectrograms[0])      # Take first batch item
        recon_audio = self.format_processor.sample_to_raw(recon_spectrograms[0])  # Take first batch item
        
        return {
            'original': original_audio,        # Now in waveform format (2, samples)
            'reconstruction': recon_audio      # Now in waveform format (2, samples)
        }