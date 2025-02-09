from typing import Optional, Union
from dataclasses import dataclass

import torch

from utils.mclt import mclt, imclt
from .format import DualDiffusionFormat, DualDiffusionFormatConfig

@dataclass
class DualMCLTFormatConfig(DualDiffusionFormatConfig):
    """Configuration for MCLT format processing
    
    Attributes:
        window_len: Length of the MCLT window
        window_fn: Window function type ("hann" etc)
        window_exponent: Window function exponent
    """
    window_len: int = 512  # MCLT window length
    window_fn: str = "hann"  # Window function type
    window_exponent: float = 0.5  # Window function exponent

class DualMCLTFormat(DualDiffusionFormat):
    """Modified Complex Lapped Transform format processor
    
    Handles conversion between raw audio and MCLT representation.
    """

    def __init__(self, config: DualMCLTFormatConfig) -> None:
        super().__init__()
        self.config = config

    def get_num_channels(self) -> tuple[int, int]:
        """Get number of input/output channels"""
        in_channels = out_channels = self.config.sample_raw_channels
        return (in_channels, out_channels)
    
    def sample_raw_crop_width(self, length: Optional[int] = None) -> int:
        """Calculate required crop width for raw samples
        
        Ensures the length is compatible with MCLT block size
        """
        block_width = self.config.window_len
        length = length or self.config.sample_raw_length
        return length // block_width // 64 * 64 * block_width + block_width
    
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:
        """Get shape of processed samples
        
        Returns shape (batch_size, channels, mclt_bins, time_steps)
        """
        _, num_output_channels = self.get_num_channels()

        crop_width = self.sample_raw_crop_width(length=length)
        num_mclt_bins = self.config.window_len // 2
        chunk_len = crop_width // num_mclt_bins - 2

        return (bsz, num_output_channels, num_mclt_bins, chunk_len)

    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert raw audio to MCLT representation
        
        Applies MCLT transform and separates magnitude/phase components
        """
        with torch.inference_mode():
            # Apply MCLT transform
            samples_mdct = mclt(raw_samples,
                              self.config.window_len,
                              self.config.window_exponent)[:, :, 1:-2, :]
            samples_mdct = samples_mdct.permute(0, 1, 3, 2)
            
            # Add random phase offset
            samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))

            # Process magnitude
            samples_mdct_abs = samples_mdct.abs()
            samples_mdct_abs_amax = samples_mdct_abs.amax(dim=(1,2,3), keepdim=True).clip(min=1e-5)
            samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs_amax).clip(min=self.config.noise_floor)
            samples_abs_ln = samples_mdct_abs.log()
            
            # Process phase
            samples_qphase1 = samples_mdct.angle().abs()
            
            # Combine components
            samples = torch.cat((samples_abs_ln, samples_qphase1), dim=1)

            # Normalize and reconstruct
            samples_mdct /= samples_mdct_abs_amax
            raw_samples = imclt(samples_mdct.permute(0, 1, 3, 2),
                              window_degree=self.config.window_exponent).real

        if return_dict:
            return {
                "samples": samples,
                "raw_samples": raw_samples,
            }
        return samples

    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert MCLT representation back to raw audio
        
        Recombines magnitude/phase and applies inverse MCLT
        """
        # Split magnitude and phase components
        samples_abs, samples_phase1 = samples.chunk(2, dim=1)
        samples_abs = samples_abs.exp()
        samples_phase = samples_phase1.cos()
        
        # Apply inverse MCLT
        raw_samples = imclt((samples_abs * samples_phase).permute(0, 1, 3, 2),
                           window_degree=self.config.window_exponent).real

        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        return raw_samples
        
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log-scaled frequencies normalized to mean=0, std=1"""
        with torch.no_grad():
            ln_freqs = torch.linspace(0, self.config.sample_rate/2, 
                                    x.shape[2] + 2, device=x.device)[1:-1].log2()
            ln_freqs = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])
            return ((ln_freqs - ln_freqs.mean()) / ln_freqs.std()).to(x.dtype)
