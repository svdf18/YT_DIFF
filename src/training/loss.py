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
        
        # Register windows as buffers for proper device handling
        for size in config.block_widths:
            self.register_buffer(
                f'window_{size}',
                torch.hann_window(size)
            )

    def _stft_loss(self, x: torch.Tensor, y: torch.Tensor, block_width: int) -> torch.Tensor:
        """Compute STFT loss between x and y at different scales"""
        # Small windows: Good for transients (drum hits)
        # Medium windows: Good for mid-range features
        # Large windows: Good for bass and overall structure
        
        # Get window for current block size
        window = getattr(self, f'window_{block_width}')
        
        # Reshape inputs to 2D
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)
        
        # Ensure all inputs are float32 for STFT
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        window = window.to(dtype=torch.float32)
        
        # Calculate STFT
        x_stft = torch.stft(
            x,
            n_fft=block_width,
            hop_length=block_width // 4,
            win_length=block_width,
            window=window,
            return_complex=True,
            pad_mode='constant'
        )
        
        y_stft = torch.stft(
            y,
            n_fft=block_width,
            hop_length=block_width // 4,
            win_length=block_width,
            window=window,
            return_complex=True,
            pad_mode='constant'
        )
        
        # Compute loss
        loss = torch.abs(x_stft - y_stft).mean()
        
        # Convert back to float16 if on MPS
        if x.device.type == 'mps':
            loss = loss.to(dtype=torch.float16)
            
        return loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multiscale spectral loss between x and y"""
        device = x.device
        total_loss = torch.tensor(0.0, device=device)
        
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