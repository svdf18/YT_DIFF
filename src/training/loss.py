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

class MultiscaleSpectralLoss:
    def __init__(self, config: MultiscaleSpectralLossConfig):
        self.config = config
        
        # Pre-compute mel filterbank if needed
        if config.mel_bands is not None:
            self.mel_fb = self._create_mel_filterbank()
        else:
            self.mel_fb = None

        # Pre-compute windows for each block size
        self.windows = {
            size: torch.hann_window(size) 
            for size in config.block_widths
        }

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix"""
        # Convert Hz to mel scale
        def hz_to_mel(f): return 2595 * np.log10(1 + f/700)
        def mel_to_hz(m): return 700 * (10**(m/2595) - 1)
        
        min_mel = hz_to_mel(self.config.freq_range[0])
        max_mel = hz_to_mel(self.config.freq_range[1])
        
        # Create mel points
        mels = torch.linspace(min_mel, max_mel, self.config.mel_bands + 2)
        freqs = torch.tensor([mel_to_hz(m) for m in mels])
        
        # Convert to FFT bins
        fft_bins = torch.floor((freqs / self.config.sample_rate) * self.config.block_widths[-1]).int()
        
        # Create filterbank matrix
        fb = torch.zeros((self.config.mel_bands, self.config.block_widths[-1]//2 + 1))
        
        for i in range(self.config.mel_bands):
            fb[i, fft_bins[i]:fft_bins[i+1]] = torch.linspace(0, 1, fft_bins[i+1]-fft_bins[i])
            fb[i, fft_bins[i+1]:fft_bins[i+2]] = torch.linspace(1, 0, fft_bins[i+2]-fft_bins[i+1])
        
        return fb

    def _stft_loss(self, x: torch.Tensor, y: torch.Tensor, block_width: int) -> torch.Tensor:
        # Ensure inputs are 2D (batch, samples)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if y.dim() > 2:
            y = y.reshape(y.shape[0], -1)
        
        # Calculate STFT with constant padding
        x_stft = torch.stft(x,
                           n_fft=block_width,
                           hop_length=block_width // 4,
                           win_length=block_width,
                           window=torch.hann_window(block_width, device=x.device),
                           return_complex=True,
                           pad_mode='constant')  # Changed from 'reflect' to 'constant'
        
        y_stft = torch.stft(y,
                           n_fft=block_width,
                           hop_length=block_width // 4,
                           win_length=block_width,
                           window=torch.hann_window(block_width, device=y.device),
                           return_complex=True,
                           pad_mode='constant')  # Changed from 'reflect' to 'constant'
        
        return torch.mean(torch.abs(x_stft - y_stft))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multiscale spectral loss between x and y"""
        total_loss = 0.0
        
        # Compute loss at each scale
        for block_width in self.config.block_widths:
            total_loss += self._stft_loss(x, y, block_width)
            
        return total_loss / len(self.config.block_widths)