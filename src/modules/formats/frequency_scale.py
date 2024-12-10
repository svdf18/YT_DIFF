from typing import Optional, Literal

import numpy as np
import torch

def _hz_to_mel(freq: float) -> float:
    """Convert Hz to Mel scale
    Formula: mel = 2595 * log10(1 + hz/700)"""
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Convert Mel scale to Hz
    Formula: hz = 700 * (10^(mel/2595) - 1)"""
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def get_mel_density(hz: torch.Tensor) -> torch.Tensor:
    """Get the density of mel filters at given frequencies"""
    return 1127. / (700. + hz)

@torch.no_grad()
def _create_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    """Create triangular filterbank for frequency scaling
    
    Args:
        all_freqs: All frequency points
        f_pts: Filter center frequencies
    
    Returns:
        Tensor: Triangular filterbank matrix
    """
    # Calculate frequency differences for filter widths
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    
    # Calculate slopes between each frequency point and filter centers
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    
    # Create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    
    # Combine slopes into final filterbank
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    
    return fb

class FrequencyScale(torch.nn.Module):
    """Frequency scaling module for audio processing
    
    Supports mel and log frequency scaling with configurable filter banks.
    """

    def __init__(
        self,
        freq_scale: Literal["mel", "log"] = "mel",
        freq_min: float = 20.0,
        freq_max: Optional[float] = None,
        sample_rate: int = 32000,
        num_stft_bins: int = 3201,
        num_filters: int = 256,
        filter_norm: Optional[Literal["slaney"]] = None,
        unscale_driver: Literal["gels", "gelsy", "gelsd", "gelss"] = "gels",
    ) -> None:
        super().__init__()

        self.freq_scale = freq_scale
        self.freq_min = freq_min
        self.freq_max = freq_max or sample_rate / 2
        self.sample_rate = sample_rate
        self.num_stft_bins = num_stft_bins
        self.num_filters = num_filters
        self.filter_norm = filter_norm
        self.unscale_driver = unscale_driver
        
        # Set scaling functions based on frequency scale type
        if freq_scale == "mel":
            self.scale_fn = _hz_to_mel
            self.unscale_fn = _mel_to_hz
        elif freq_scale == "log":
            self.scale_fn = np.log2
            self.unscale_fn = torch.exp2
        else:
            raise ValueError(f"Unknown frequency scale: {freq_scale}")
        
        # Create and register filterbank
        self.register_buffer("filters", self.get_filters())
        
        # Validate filterbank
        if (self.filters.max(dim=0).values == 0.0).any():
            print("Warning: At least one FrequencyScale filterbank has all zero values.")

    @torch.no_grad()
    def scale(self, specgram: torch.Tensor) -> torch.Tensor:
        """Apply frequency scaling to spectrogram"""
        return torch.matmul(specgram.transpose(-1, -2), self.filters).transpose(-1, -2)
    
    @torch.no_grad()
    def unscale(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Inverse frequency scaling"""
        original_shape = spectrogram.size()
        spectrogram = spectrogram.view(-1, original_shape[-2], original_shape[-1])
        
        # Use least squares to approximate inverse
        unscaled = torch.relu(torch.linalg.lstsq(
            self.filters.transpose(-1, -2)[None],
            spectrogram, 
            driver=self.unscale_driver
        ).solution)
        
        return unscaled.view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))
    
    @torch.no_grad()
    def get_unscaled(self, num_points: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get unscaled frequency points"""
        scaled_freqs = torch.linspace(
            self.scale_fn(self.freq_min), 
            self.scale_fn(self.freq_max), 
            num_points, 
            device=device
        )
        return self.unscale_fn(scaled_freqs)
    
    @torch.no_grad()
    def get_filters(self) -> torch.Tensor:
        """Create filterbank matrix"""
        stft_freqs = torch.linspace(0, self.sample_rate / 2, self.num_stft_bins)
        unscaled_freqs = self.get_unscaled(self.num_filters + 2)
        filters = _create_triangular_filterbank(stft_freqs, unscaled_freqs)

        if self.filter_norm == "slaney":
            # Slaney-style mel scaling for constant energy per channel
            enorm = 2. / (unscaled_freqs[2:self.num_filters+2] - unscaled_freqs[:self.num_filters])
            filters *= enorm.unsqueeze(0)

        return filters