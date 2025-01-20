import torch
import torchaudio
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from .format import DualDiffusionFormat, DualDiffusionFormatConfig

@dataclass
class SimpleSpectrogramConfig(DualDiffusionFormatConfig):
    """Simplified configuration for spectrogram processing"""
    # Audio parameters (from format.json)
    sample_rate: int = 32000
    sample_raw_channels: int = 2
    sample_raw_length: int = 1440000
    noise_floor: float = 2e-05
    
    # FFT parameters (derived from format.json)
    n_fft: int = 2048
    step_size_ms: int = 8  # hop_length_ms
    window_duration_ms: int = 64  # win_length_ms
    padded_duration_ms: int = 64
    
    # Window parameters (from format.json)
    window_exponent: float = 32.0
    window_periodic: bool = True
    
    # Spectrogram parameters (from format.json)
    num_frequencies: int = 80  # Changed from 256 to match format.json
    abs_exponent: float = 0.25
    t_scale: Optional[float] = None
    
    # Frequency scale parameters (from format.json)
    freq_scale_type: str = "mel"
    min_frequency: int = 20
    max_frequency: int = 20000
    freq_scale_norm: Optional[float] = None
    
    # Griffin-Lim parameters (from format.json)
    num_fgla_iters: int = 200  # griffin_lim_iters
    fgla_momentum: float = 0.99  # momentum
    stereo_coherence: float = 0.67
    
    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms * self.sample_rate / 1000)
    
    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms * self.sample_rate / 1000)

def mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Convert mel scale to Hz"""
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """Convert Hz to mel scale"""
    return 2595.0 * torch.log10(1.0 + (freq / 700.0))

class SimpleSpectrogramFormat(DualDiffusionFormat):
    """Simplified spectrogram format based on working reconstruction code"""
    
    def __init__(self, config: SimpleSpectrogramConfig):
        super().__init__()
        self.config = config
        
        # Create powered window once
        self.window = self._create_window()
        
        # Initialize transforms
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            window_fn=lambda size: self.window,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True
        )
        
        # Create mel filterbank
        if config.freq_scale_type == "mel":
            mel_fb, mel_fb_inverse = self._create_mel_filterbank()
            # Register as buffers so they move with the module
            self.register_buffer('mel_fb', mel_fb)
            self.register_buffer('mel_fb_inverse', mel_fb_inverse)
        
        # Initialize Griffin-Lim
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            window_fn=lambda size: self.window,
            power=1.0,
            n_iter=config.num_fgla_iters,
            momentum=config.fgla_momentum,
            length=None,
            rand_init=False
        )
    
    def _create_window(self) -> torch.Tensor:
        """Create a powered Hann window"""
        window = torch.hann_window(
            self.config.win_length, 
            periodic=self.config.window_periodic
        )
        return window ** self.config.window_exponent
    
    def _create_mel_filterbank(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create mel filterbank and its inverse"""
        # Get frequencies in Hz
        fft_freqs = torch.linspace(0, self.config.sample_rate // 2, 256)  # Match VAE output
        
        # Convert to mel scale
        mel_min = hz_to_mel(torch.tensor(self.config.min_frequency))
        mel_max = hz_to_mel(torch.tensor(self.config.max_frequency))
        mel_points = torch.linspace(mel_min, mel_max, self.config.num_frequencies + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Create filterbank matrix [num_mel, n_freq]
        fbank = torch.zeros((self.config.num_frequencies, 256))
        
        # Create triangular filters
        for i in range(self.config.num_frequencies):
            lower = hz_points[i]
            peak = hz_points[i + 1]
            upper = hz_points[i + 2]
            
            # Lower slope
            lower_mask = (fft_freqs >= lower) & (fft_freqs <= peak)
            fbank[i, lower_mask] = (fft_freqs[lower_mask] - lower) / (peak - lower)
            
            # Upper slope
            upper_mask = (fft_freqs >= peak) & (fft_freqs <= upper)
            fbank[i, upper_mask] = (upper - fft_freqs[upper_mask]) / (upper - peak)
        
        # Normalize each filter
        fbank = fbank / (fbank.sum(dim=1, keepdim=True) + 1e-8)
        
        # Create pseudo-inverse for reconstruction
        fbank_inverse = torch.pinverse(fbank)  # [n_freq, num_mel]
        
        return fbank, fbank_inverse
    
    def _apply_frequency_scaling(self, spec: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply frequency scaling (mel or linear)"""
        if self.config.freq_scale_type == "mel":
            # 1. Debug input state
            batch, channels, freq, time = spec.shape
            print(f"\nFrequency scaling debug:")
            print(f"1. Input shape: {spec.shape}")
            print(f"   Expected freq bins: {'256 (linear)' if inverse else '80 (mel)'}")
            print(f"   Mel FB shape: {self.mel_fb.shape}")
            
            # 2. Prepare for matrix multiplication
            spec_flat = spec.permute(0, 1, 3, 2).reshape(-1, freq)  # [batch*channels*time, freq]
            print(f"2. Flattened shape: {spec_flat.shape}")
            
            # 3. Apply frequency scaling
            if inverse:
                # Going from mel (80) to linear (256)
                spec_scaled = torch.matmul(spec_flat, self.mel_fb_inverse)
                new_freq = self.mel_fb_inverse.shape[1]  # Should be 256
            else:
                # Going from linear (256) to mel (80)
                spec_scaled = torch.matmul(spec_flat, self.mel_fb.t())
                new_freq = self.mel_fb.shape[0]  # Should be 80
            
            print(f"3. After scaling shape: {spec_scaled.shape}")
            print(f"   New frequency bins: {new_freq}")
            
            # 4. Restore original dimensions
            spec_final = spec_scaled.reshape(batch, channels, time, new_freq).permute(0, 1, 3, 2)
            print(f"4. Final shape: {spec_final.shape}")
            
            return spec_final
        
        return spec
    
    def get_num_channels(self) -> tuple[int, int]:
        """Get number of input/output channels"""
        return (self.config.sample_raw_channels, self.config.sample_raw_channels)
    
    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor,
                     return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert raw audio to spectrogram"""
        # Normalize audio if configured
        if self.config.normalize_audio:
            raw_samples = raw_samples / raw_samples.abs().max()
        
        # Process each channel
        specs = []
        for channel in range(raw_samples.shape[1]):
            # Get spectrogram
            spec = self.spectrogram(raw_samples[:, channel])
            # Take magnitude
            spec = spec.abs()
            # Apply frequency scaling
            spec = self._apply_frequency_scaling(spec)
            # Apply power law
            spec = spec ** self.config.abs_exponent
            specs.append(spec)
        
        # Stack channels
        samples = torch.stack(specs, dim=1)
        
        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        return samples
    
    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor,
                     return_dict: bool = False) -> Union[torch.Tensor, dict]:
        """Convert spectrogram back to raw audio"""
        try:
            print(f"\nSample to raw conversion:")
            print(f"1. Input shape: {samples.shape}")
            
            # Process each channel
            audio_channels = []
            for channel in range(samples.shape[1]):
                # Get channel data
                spec = samples[:, channel]  # [batch, freq, time]
                print(f"\nChannel {channel}:")
                print(f"2. Channel shape: {spec.shape}")
                
                # Unscale from power law
                spec = spec ** (1 / self.config.abs_exponent)
                
                # Add channel dim for scaling
                spec = spec.unsqueeze(1)  # [batch, 1, freq, time]
                print(f"3. Pre-scaling shape: {spec.shape}")
                
                # Apply inverse frequency scaling to get back to linear frequencies
                spec = self._apply_frequency_scaling(spec, inverse=True)
                print(f"4. Post-scaling shape: {spec.shape}")
                
                # Remove channel dim and ensure positive
                spec = spec.squeeze(1).abs()
                print(f"5. Pre-Griffin-Lim shape: {spec.shape}")
                
                # Pad with zeros to match expected frequency dimension if needed
                expected_freq = self.config.n_fft // 2 + 1
                if spec.shape[1] < expected_freq:
                    pad_size = expected_freq - spec.shape[1]
                    spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_size))
                print(f"6. After padding shape: {spec.shape}")
                
                # Reconstruct audio
                audio = self.griffin_lim(spec)
                
                # Normalize audio to [-1, 1] range
                audio = audio / (audio.abs().max() + 1e-8)
                
                # Convert to float32 and ensure it's contiguous
                audio = audio.to(torch.float32).contiguous()
                
                audio_channels.append(audio)
                print(f"7. Audio output shape: {audio.shape}")
            
            # Stack channels
            raw_samples = torch.stack(audio_channels, dim=1)
            print(f"\nFinal output shape: {raw_samples.shape}")
            
            # Ensure audio is in CPU, contiguous, and proper format for saving
            raw_samples = raw_samples.cpu().contiguous()
            
            # Transpose to [batch, time, channels] format for soundfile
            raw_samples = raw_samples.transpose(1, 2)
            
            if return_dict:
                return {"samples": samples, "raw_samples": raw_samples}
            return raw_samples
            
        except Exception as e:
            print(f"\nError in sample_to_raw:")
            print(f"Input shape: {samples.shape}")
            print(f"Error: {str(e)}")
            raise e

    # Implement required abstract methods
    def get_ln_freqs(self) -> int:
        """Get number of frequency bins"""
        return self.config.num_frequencies
    
    def get_sample_shape(self) -> tuple[int, ...]:
        """Get shape of processed samples"""
        return (self.config.sample_raw_channels, self.config.num_frequencies, -1)
    
    def sample_raw_crop_width(self) -> int:
        """Get width for cropping raw samples"""
        return self.config.sample_raw_length