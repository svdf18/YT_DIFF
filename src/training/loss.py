"""
This file implements loss functions for training the YT_DIFF model.

Key components:

1. Multiscale Spectral Loss:
   - Computes spectral loss across multiple window sizes
   - Uses STFT to analyze frequency content
   - Handles stereo audio with separation weighting
   
2. Loss Configuration:
   - MultiscaleSpectralLoss1DConfig defines parameters
   - Controls window sizes, overlap, weighting etc.
   - Configures spectral analysis settings

3. Loss Calculation:
   - Compares generated and target spectrograms
   - Applies appropriate weighting and scaling
   - Handles mono and stereo cases differently

4. Helper Functions:
   - STFT and mel-scale conversions
   - Window function generation
   - Loss aggregation across scales

Enhanced spectral loss implementation with:
1. Mixed magnitude/phase loss
2. Mel-weighted phase importance
3. 2D spectral analysis for spectrograms
4. Stereo separation handling
"""

from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

@dataclass
class MultiscaleSpectralLossConfig:
    """Configuration for multiscale spectral loss"""
    block_widths: List[int] = (8, 16, 32, 64)  # Window sizes for STFT
    block_overlap: int = 8                      # Overlap between windows
    mel_bands: Optional[int] = None            # Number of mel bands (None = linear)
    freq_range: Tuple[float, float] = (20, 20000)  # Frequency range in Hz
    sample_rate: int = 32000                   # Audio sample rate
    stereo_weight: float = 0.5                 # Weight for stereo separation loss

class MultiscaleSpectralLoss(torch.nn.Module):
    def __init__(self, config: MultiscaleSpectralLossConfig):
        super().__init__()
        self.config = config
        
        # Create windows for each block width
        self.windows = {}
        for block_width in config.block_widths:
            window = torch.hann_window(block_width)
            self.register_buffer(f'window_{block_width}', window)
    
    def _stft_loss(self, x: torch.Tensor, y: torch.Tensor, block_width: int) -> torch.Tensor:
        """Compute STFT loss for a specific block width"""
        # Get the correct window for this block width
        window = getattr(self, f'window_{block_width}')
        
        # Process each channel separately
        total_loss = 0.0
        batch_size, channels, freq, time = x.shape
        
        for c in range(channels):
            # Get channel data
            x_ch = x[:, c].reshape(batch_size, freq * time)
            y_ch = y[:, c].reshape(batch_size, freq * time)
            
            # Compute STFTs
            x_stft = torch.stft(
                x_ch,
                n_fft=block_width,
                hop_length=block_width // self.config.block_overlap,
                win_length=block_width,
                window=window,
                return_complex=True,
                pad_mode='constant'
            )
            
            y_stft = torch.stft(
                y_ch,
                n_fft=block_width,
                hop_length=block_width // self.config.block_overlap,
                win_length=block_width,
                window=window,
                return_complex=True,
                pad_mode='constant'
            )
            
            # Compute magnitude spectrograms
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            
            # Compute log magnitude spectrograms
            x_log_mag = torch.log1p(x_mag)
            y_log_mag = torch.log1p(y_mag)
            
            # Compute L1 loss between log magnitudes
            channel_loss = torch.mean(torch.abs(x_log_mag - y_log_mag))
            
            # Weight stereo channels if configured
            if c == 1:  # Right channel
                channel_loss = channel_loss * self.config.stereo_weight
                
            total_loss = total_loss + channel_loss
        
        return total_loss / channels
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale spectral loss"""
        # Validate input shapes
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=x.device)
        
        # Compute loss at each scale
        for block_width in self.config.block_widths:
            total_loss = total_loss + self._stft_loss(x, y, block_width)
        
        return total_loss / len(self.config.block_widths)

    # Alias call to forward for backward compatibility
    __call__ = forward

    def _handle_stereo(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split stereo into sum/difference channels"""
        if x.shape[1] == 2:
            return (
                x[:, 0] + x[:, 1],  # Sum (mid)
                x[:, 0] - x[:, 1]   # Difference (side)
            )
        return x, None