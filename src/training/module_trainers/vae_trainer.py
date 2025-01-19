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
from typing import Optional, Dict, Any

import torch
from accelerate import Accelerator

# Import device utilities
from src.utils.device import device, get_device, init_training

from src.modules.vaes.vae_edm2 import DualDiffusionVAE_EDM2
from src.modules.formats.spectrogram import SpectrogramFormat
from src.training.loss import MultiscaleSpectralLoss, MultiscaleSpectralLossConfig

@dataclass
class VAETrainerConfig:
    # EDM2-specific settings
    target_snr: float = 20.0
    res_balance: float = 0.4
    attn_balance: float = 0.4
    
    # Loss weights
    channel_kl_loss_weight: float = 0.05  # Reduced from 0.1
    recon_loss_weight: float = 1.0  # Increased from 0.1
    
    # Spectral loss settings
    block_widths: tuple[int, ...] = (16, 32, 64, 128)  # Larger windows
    block_overlap: int = 8
    mel_bands: int = 80
    min_frequency: int = 20
    max_frequency: int = 20000
    sample_rate: int = 32000
    stereo_weight: float = 0.8  # Increased from 0.67
    
    # Training settings
    gradient_clip_val: float = 1.0
    batch_size: int = 16  # Match training config
    
    def validate_batch_size(self, batch_size: int):
        if batch_size != self.batch_size:
            print(f"Warning: Batch size mismatch! Expected {self.batch_size}, got {batch_size}")

class VAETrainer:
    def __init__(self, model: torch.nn.Module, 
                 config: VAETrainerConfig,
                 format_processor: SpectrogramFormat,
                 accelerator: Optional[Accelerator] = None) -> None:
        """Initialize trainer with EDM2 features and simplified loss"""
        super().__init__()
        self.model = model
        self.config = config
        self.format_processor = format_processor
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator else 'cpu'
        
        # Initialize simplified spectral loss
        spectral_loss_config = MultiscaleSpectralLossConfig(
            block_widths=list(config.block_widths),
            block_overlap=config.block_overlap,
            mel_bands=None,  # Simplified version doesn't use mel bands
            freq_range=(config.min_frequency, config.max_frequency),
            sample_rate=config.sample_rate,
            stereo_weight=config.stereo_weight
        )
        self.spectral_loss = MultiscaleSpectralLoss(spectral_loss_config).to(self.device)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step with enhanced debugging"""
        try:
            # Add batch validation
            spectrograms = batch['audio'].to(self.device)
            batch_size = spectrograms.shape[0]
            
            print(f"\nBatch Processing:")
            print(f"Current batch size: {batch_size}")
            print(f"Expected batch size: {self.config.batch_size}")
            
            if batch_size != self.config.batch_size:
                print(f"Warning: Unexpected batch size! Got {batch_size}, expected {self.config.batch_size}")
            
            # 1. Input validation
            print("\nInput validation:")
            print(f"Batch keys: {batch.keys()}")
            print(f"Input spectrograms shape: {spectrograms.shape}")
            print(f"Input dtype: {spectrograms.dtype}")
            print(f"Input device: {spectrograms.device}")
            
            with self.accelerator.autocast():
                # 2. Encode
                z = self.model.encode(spectrograms)
                print("\nLatent space:")
                print(f"z.mean shape: {z.mean.shape}")
                print(f"z.logvar shape: {z.logvar.shape if hasattr(z, 'logvar') else 'N/A'}")
                
                # 3. Sample
                z_sample = z.sample()
                print(f"Sampled z shape: {z_sample.shape}")
                
                # 4. Decode
                recon = self.model.decode(z_sample)
                print("\nReconstruction:")
                print(f"Reconstruction shape: {recon.shape}")
                print(f"Reconstruction dtype: {recon.dtype}")
                print(f"Reconstruction device: {recon.device}")
                
                # 5. Verify shapes match for loss
                if recon.shape != spectrograms.shape:
                    print("\nShape mismatch!")
                    print(f"Expected shape: {spectrograms.shape}")
                    print(f"Got shape: {recon.shape}")
                    raise ValueError(f"Shape mismatch between input and reconstruction")
                
                # 6. Compute losses
                recon_loss = self.spectral_loss(recon, spectrograms)
                kl_loss = z.kl() * self.config.channel_kl_loss_weight
                loss = recon_loss * self.config.recon_loss_weight + kl_loss
                
                print("\nLoss values:")
                print(f"Reconstruction loss: {recon_loss:.3f}")
                print(f"KL loss: {kl_loss:.3f}")
                print(f"Total loss: {loss:.3f}")
                
            return {
                'loss': loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'target_snr': self.model.get_target_snr()
            }
            
        except Exception as e:
            print("\nError in training step:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            raise e

    @torch.no_grad()
    def generate_samples(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate samples with enhanced shape handling"""
        try:
            self.model.eval()
            print("\nGenerate samples:")
            
            # 1. Input processing
            spectrograms = batch['audio'].to(self.device)
            print(f"Input shape: {spectrograms.shape}")
            
            # 2. VAE forward pass
            z = self.model.encode(spectrograms)
            print(f"Encoded shape: {z.mean.shape}")
            
            recon = self.model.decode(z.sample())
            print(f"Reconstruction shape: {recon.shape}")
            
            # 3. Format conversion with shape validation
            print("\nFormat conversion:")
            print(f"Pre-conversion shape: {spectrograms.shape}")
            
            # Ensure correct shape before conversion
            if spectrograms.dim() == 4:  # [batch, channels, freq, time]
                batch_size, channels, freq, time = spectrograms.shape
                # Reshape if needed
                spectrograms = spectrograms.reshape(batch_size * channels, freq, time)
                recon = recon.reshape(batch_size * channels, freq, time)
            
            print(f"Conversion input shapes - Original: {spectrograms.shape}, Recon: {recon.shape}")
            
            # Convert to audio
            original_audio = self.format_processor.sample_to_raw(spectrograms)
            recon_audio = self.format_processor.sample_to_raw(recon)
            
            print(f"Output audio shapes - Original: {original_audio.shape}, Recon: {recon_audio.shape}")
            
            # Handle metadata more safely
            metadata = {
                'paths': batch['paths'][0] if isinstance(batch['paths'], (list, tuple)) else batch['paths'],
                'categories': batch['categories'][0] if isinstance(batch['categories'], (list, tuple)) else batch['categories'],
                'labels': batch['labels'][0] if isinstance(batch['labels'], (list, tuple)) else batch['labels']
            }
            
            self.model.train()
            return {
                'original': original_audio,
                'reconstruction': recon_audio,
                'target_snr': self.model.get_target_snr(),
                'metadata': metadata
            }
            
        except Exception as e:
            print("\nError in generate_samples:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nShape debug info:")
            print(f"Last known spectrograms shape: {spectrograms.shape}")
            print(f"Last known recon shape: {recon.shape}")
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            raise e