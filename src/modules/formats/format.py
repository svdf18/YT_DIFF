from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig

@dataclass
class DualDiffusionFormatConfig(DualDiffusionModuleConfig):
    """Base configuration for audio format processing
    
    Attributes:
        noise_floor: Minimum value for audio samples to prevent log(0)
        sample_rate: Audio sampling rate in Hz
        sample_raw_channels: Number of audio channels (2 for stereo)
        sample_raw_length: Default length of audio samples in samples
        t_scale: Optional time scaling factor for processing
    """
    noise_floor: float = 2e-5  # -94dB, typical noise floor for 16-bit audio
    sample_rate: int = 32000   # 32kHz sampling rate
    sample_raw_channels: int = 2  # Stereo by default
    sample_raw_length: int = 1057570  # ~33 seconds at 32kHz
    t_scale: Optional[float] = None  # Time scaling factor, if needed

class DualDiffusionFormat(DualDiffusionModule, ABC):
    """Abstract base class for audio format processing
    
    This class defines the interface for converting between raw audio and the model's
    internal representation (typically spectrograms). All audio format processors must
    implement these methods.
    """

    module_name: str = "format"  # Module identifier
    has_trainable_parameters: bool = False  # Format modules are typically parameter-free
    supports_half_precision: bool = False  # Complex numbers don't support half precision
    supports_compile: bool = False  # torch.compile doesn't support complex operators
    
    @abstractmethod
    def get_num_channels(self) -> tuple[int, int]:
        """Returns the number of input and output channels
        
        Returns:
            tuple: (input_channels, output_channels)
            For stereo audio, typically (2, 2)
        """
        pass
    
    @abstractmethod
    def sample_raw_crop_width(self, length: Optional[int] = None) -> int:
        """Calculate required crop width for raw audio samples
        
        Args:
            length: Optional target length in samples
                   If None, uses config.sample_raw_length
        
        Returns:
            int: Number of samples needed for processing window
        """
        pass
    
    @abstractmethod
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:
        """Get shape of processed samples
        
        Args:
            bsz: Batch size
            length: Optional target length in samples
        
        Returns:
            tuple: Shape of processed samples (B, C, F, T)
                  B: batch size
                  C: channels
                  F: frequency bins
                  T: time steps
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert raw audio to model format
        
        Args:
            raw_samples: Raw audio tensor (B, C, T)
            return_dict: If True, returns dict with additional info
        
        Returns:
            Union[Tensor, dict]: Processed samples in model format
                                If return_dict, includes raw samples
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert model format back to raw audio
        
        Args:
            samples: Processed samples in model format
            return_dict: If True, returns dict with additional info
        
        Returns:
            Union[Tensor, dict]: Reconstructed raw audio
                                If return_dict, includes processed samples
        """
        pass
    
    @abstractmethod
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log-scaled frequencies for the given tensor
        
        Args:
            x: Input tensor to get frequencies for
        
        Returns:
            Tensor: Log-scaled frequencies, normalized to mean=0, std=1
        """
        pass

    def compile(self, **kwargs) -> None:
        """Compile the format processing if supported
        
        Note: Currently disabled by default as torch.compile
        doesn't support complex number operations needed for
        audio processing.
        
        Args:
            **kwargs: Compilation arguments passed to torch.compile
        """
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.raw_to_sample = torch.compile(self.raw_to_sample, **kwargs)
            self.sample_to_raw = torch.compile(self.sample_to_raw, **kwargs)