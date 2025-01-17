from typing import Optional, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm
from src.utils.device import device  # Import the device utility

def _get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Get corresponding complex dtype for real dtype
    
    Args:
        real_dtype: Input real dtype
    
    Returns:
        torch.dtype: Corresponding complex dtype
    
    Raises:
        ValueError: If no matching complex dtype exists
    """
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")

@torch.inference_mode()
def griffinlim(
    specgram: Tensor,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    momentum: float,
    length: Optional[int],
    rand_init: bool,
    stereo: bool,
    stereo_coherence: float,
    manual_init: Optional[Tensor] = None,
    show_tqdm: bool = True,
) -> Tensor:
    """Griffin-Lim algorithm for phase reconstruction"""
    # Move everything to CPU for complex operations
    original_device = specgram.device
    specgram = specgram.cpu()
    window = window.cpu()
    if manual_init is not None:
        manual_init = manual_init.cpu()

    if not 0 <= momentum < 1:
        raise ValueError(f"momentum must be in range [0, 1). Found: {momentum}")
    momentum = momentum / (1 + momentum)

    # Reshape spectrogram for processing
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    # Handle stereo processing
    if stereo:
        merged_specgram = ((specgram[0::2] + specgram[1::2]) / 2).repeat_interleave(2, dim=0)
    
    # Initialize phase angles
    if manual_init is not None:
        angles = manual_init.reshape(specgram.shape)
    else:
        init_shape = (1,) + tuple(specgram.shape[1:])
        if rand_init:
            angles = torch.randn(init_shape, dtype=_get_complex_dtype(specgram.dtype), device='cpu')
        else:
            angles = torch.full(init_shape, 1, dtype=_get_complex_dtype(specgram.dtype), device='cpu')
        
    # Previous iteration state for momentum
    tprev = torch.tensor(0.0, dtype=specgram.dtype, device='cpu')

    # Main iteration loop
    progress_bar = tqdm(total=n_iter) if show_tqdm else None
    for i in range(n_iter):
        # Handle stereo coherence
        if stereo:
            t = i / n_iter - stereo_coherence
            if t > 0:
                interp_specgram = torch.lerp(merged_specgram, specgram, t)
            else:
                interp_specgram = merged_specgram
        else:
            interp_specgram = specgram

        # Inverse STFT
        inverse = torch.istft(
            angles * interp_specgram, 
            n_fft=n_fft, 
            hop_length=hop_length,
            win_length=win_length, 
            window=window, 
            length=length
        )

        # Forward STFT
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Update angles with momentum
        angles = rebuilt
        if momentum:
            angles.sub_(tprev, alpha=momentum)
        angles = angles.div(angles.abs().add_(1e-16))

        tprev = rebuilt
        if progress_bar is not None:
            progress_bar.update(1)

    # Final reconstruction
    waveform = torch.istft(
        angles * specgram, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window, 
        length=length
    )

    if progress_bar is not None:
        progress_bar.close()

    # Move result back to original device
    return waveform.reshape(shape[:-2] + waveform.shape[-1:]).to(original_device)

class PhaseRecovery(torch.nn.Module):
    """Phase recovery module using Griffin-Lim algorithm"""

    def __init__(
        self,
        n_fft: int,
        n_fgla_iter: int = 200,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        momentum: float = 0.99,
        length: Optional[int] = None,
        rand_init: bool = True,
        stereo: bool = True,
        stereo_coherence: float = 0.67,
    ) -> None:
        super().__init__()

        if not (0 <= momentum < 1):
            raise ValueError(f"momentum must be in range [0, 1). Found: {momentum}")

        self.n_fft = n_fft
        self.n_fgla_iter = n_fgla_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.length = length
        self.momentum = momentum
        self.rand_init = rand_init
        self.stereo = stereo
        self.stereo_coherence = stereo_coherence

        # Create and register window - move to device from utils
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window.to(device), persistent=False)

    @torch.inference_mode()
    def forward(self, specgram: Tensor, n_fgla_iter: Optional[int] = None, show_tqdm: bool = True) -> Tensor:
        """Apply phase recovery to spectrogram"""
        n_fgla_iter = n_fgla_iter or self.n_fgla_iter
        original_device = specgram.device

        if self.n_fgla_iter > 0:
            # Move everything to CPU for processing
            specgram = specgram.cpu()
            self.window = self.window.cpu()
            
            wave = griffinlim(
                specgram,
                self.window,
                self.n_fft,
                self.hop_length,
                self.win_length,
                n_fgla_iter,
                self.momentum,
                self.length,
                self.rand_init,
                self.stereo,
                self.stereo_coherence,
                manual_init=None,
                show_tqdm=show_tqdm,
            )
            
            # Move result back to original device
            return wave.to(original_device)
        else:
            # Direct inverse if no iterations requested
            wave_shape = specgram.size()
            wave = torch.istft(
                specgram.reshape([-1] + list(wave_shape[-2:])),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.cpu(),  # Ensure window is on CPU
                length=self.length,
            ).reshape(wave_shape[:-2] + (-1,))
            
            return wave.to(original_device)