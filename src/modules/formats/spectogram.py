import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, Literal

from .format import DualDiffusionFormat, DualDiffusionFormatConfig
from .phase_recovery import PhaseRecovery

@dataclass
class SpectrogramFormatConfig(DualDiffusionFormatConfig):
    """Configuration for spectrogram processing
    
    Attributes:
        abs_exponent: Exponent applied to spectrogram magnitudes
        sample_rate: Audio sample rate in Hz
        step_size_ms: Time step between FFT windows in milliseconds
        window_duration_ms: Duration of analysis window in milliseconds
        padded_duration_ms: Total padded window duration in milliseconds
        window_exponent: Power to raise window function to
        window_periodic: Whether to use periodic window
        num_frequencies: Number of frequency bins
        min_frequency: Minimum frequency in Hz
        max_frequency: Maximum frequency in Hz
    """
    abs_exponent: float = 0.25
    
    # FFT parameters
    sample_rate: int = 32000
    step_size_ms: int = 8
    
    # Window settings
    window_duration_ms: int = 200
    padded_duration_ms: int = 200
    window_exponent: float = 32
    window_periodic: bool = True
    
    # Frequency scale parameters
    num_frequencies: int = 256
    min_frequency: int = 20
    max_frequency: int = 16000
    
    # Phase recovery parameters
    num_griffin_lim_iters: int = 200
    momentum: float = 0.99
    stereo_coherence: float = 0.67

    @property
    def stereo(self) -> bool:
        return self.sample_raw_channels == 2
    
    @property
    def num_stft_bins(self) -> int:
        return self.padded_length // 2 + 1
    
    @property
    def padded_length(self) -> int:
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms / 1000.0 * self.sample_rate)

class SpectrogramConverter(torch.nn.Module):
    """Handles conversion between audio and spectrogram representations"""

    @staticmethod
    def hann_power_window(window_length: int, periodic: bool = True, *, dtype: torch.dtype = None,
                         layout: torch.layout = torch.strided, device: torch.device = None,
                         requires_grad: bool = False, exponent: float = 1.) -> torch.Tensor:
        """Create a powered Hann window"""
        return torch.hann_window(window_length, periodic=periodic, dtype=dtype,
                               layout=layout, device=device, requires_grad=requires_grad) ** exponent

    def __init__(self, config: SpectrogramFormatConfig):
        super().__init__()
        self.config = config
        
        window_args = {
            "exponent": config.window_exponent,
            "periodic": config.window_periodic,
        }

        # Initialize spectrogram transform
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=config.padded_length,
            win_length=config.win_length,
            hop_length=config.hop_length,
            pad=0,
            window_fn=self.hann_power_window,
            power=None,
            normalized=False,
            wkwargs=window_args,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        # Initialize phase recovery for inverse transform
        self.inverse_spectrogram_func = PhaseRecovery(
            n_fft=config.padded_length,
            n_iter=config.num_griffin_lim_iters,
            win_length=config.win_length,
            hop_length=config.hop_length,
            window_fn=self.hann_power_window,
            wkwargs=window_args,
            momentum=config.momentum,
            length=None,
            rand_init=False,
            stereo=config.stereo,
            stereo_coherence=config.stereo_coherence
        )

    def get_spectrogram_shape(self, audio_shape: torch.Size) -> torch.Size:
        """Calculate output spectrogram shape for given audio input shape"""
        num_frames = 1 + (audio_shape[-1] + self.config.padded_length - self.config.win_length) // self.config.hop_length
        return torch.Size(audio_shape[:-1] + (self.config.num_frequencies, num_frames))

    @torch.inference_mode()
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to spectrogram representation"""
        spectrogram_complex = self.spectrogram_func(audio)
        return spectrogram_complex.abs() ** self.config.abs_exponent

    @torch.inference_mode()
    def spectrogram_to_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram back to audio using phase recovery"""
        amplitudes = spectrogram ** (1 / self.config.abs_exponent)
        return self.inverse_spectrogram_func(amplitudes)

class SpectrogramFormat(DualDiffusionFormat):
    """Main spectrogram format handler"""

    def __init__(self, config: SpectrogramFormatConfig):
        super().__init__()
        self.config = config
        self.converter = SpectrogramConverter(config)
    
    def get_num_channels(self) -> tuple[int, int]:
        """Get number of input/output channels"""
        in_channels = out_channels = self.config.sample_raw_channels
        return (in_channels, out_channels)
    
    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor,
                     return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert raw audio to normalized spectrogram"""
        samples = self.converter.audio_to_spectrogram(raw_samples)
        samples /= samples.std(dim=(1,2,3), keepdim=True).clip(min=self.config.noise_floor)

        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        return samples

    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor,
                     return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert spectrogram back to raw audio"""
        raw_samples = self.converter.spectrogram_to_audio(samples.clip(min=0))

        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        return raw_samples