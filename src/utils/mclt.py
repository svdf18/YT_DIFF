from typing import Optional

"""
This file contains the implementation of the MCLT (Modified Chirp-Like Transform) for audio processing.

The MCLT is a signal processing transform that decomposes audio signals into frequency components 
while preserving phase information. The key steps are:

1. Windowing:
   - The input signal is divided into overlapping frames
   - Each frame is multiplied by a window function (e.g. Kaiser, Hann) to reduce spectral leakage
   
2. Modified DCT/DST:
   - The windowed frames undergo a modified discrete cosine transform (MDCT)
   - A modified discrete sine transform (MDST) is also applied
   - These provide the real and imaginary components respectively

3. Frequency Analysis:
   - The MDCT/MDST coefficients represent frequency content
   - Phase information is preserved through the complex representation
   - Allows for high quality reconstruction of the original signal

4. Inverse Transform:
   - The inverse MCLT reconstructs the time domain signal
   - Uses overlap-add synthesis with the analysis window
   - Provides perfect reconstruction when using appropriate windows

The MCLT is particularly useful for audio processing tasks like:
- Compression and coding
- Time-frequency analysis
- Audio effects and modifications
- Feature extraction for machine learning

This implementation provides the core MCLT functionality along with various window 
functions optimized for different applications.
"""

import torch

class WindowFunction:
    """Window functions for signal processing
    
    This class provides various window functions used in signal processing
    to reduce spectral leakage when performing frequency analysis.
    Each window function has different tradeoffs between main lobe width
    and side lobe levels.
    """

    @staticmethod
    def hann(window_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Hann window function
        
        Classic cosine-based window with good all-around properties.
        Formula: w[n] = 0.5 * (1 - cos(2π * n/N))
        
        Args:
            window_len: Length of the window in samples
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Hann window of specified length
        """
        n = torch.arange(window_len, device=device) / window_len
        return (0.5 - 0.5 * torch.cos(2 * torch.pi * n))

    @staticmethod
    def kaiser(window_len: int, beta: float = 4*torch.pi, device: Optional[torch.device] = None) -> torch.Tensor:
        """Kaiser window function
        
        Parameterized window function based on modified Bessel function.
        Provides excellent sidelobe control through beta parameter.
        
        Args:
            window_len: Length of the window in samples
            beta: Shape parameter controlling sidelobe level
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Kaiser window of specified length
        """
        alpha = (window_len - 1) / 2
        n = torch.arange(window_len, device=device)
        return (torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha).square()))
                / torch.special.i0(torch.tensor(beta)))

    @staticmethod
    def kaiser_derived(window_len:int , beta: float = 4*torch.pi, device: Optional[torch.device] = None) -> torch.Tensor:
        """Derived Kaiser window function
        
        Modified Kaiser window with improved properties for certain applications.
        Constructed by taking cumulative sum of Kaiser window and symmetrizing.
        
        Args:
            window_len: Length of the window in samples
            beta: Shape parameter controlling sidelobe level
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Derived Kaiser window of specified length
        """
        kaiserw = WindowFunction.kaiser(window_len // 2 + 1, beta, device)
        csum = torch.cumsum(kaiserw, dim=0)
        halfw = torch.sqrt(csum[:-1] / csum[-1])

        w = torch.zeros(window_len, device=device)
        w[:window_len//2] = halfw
        w[-window_len//2:] = halfw.flip(0)

        return w

    @staticmethod
    def hann_poisson(window_len: int, alpha: float = 2, device: Optional[torch.device] = None) -> torch.Tensor:
        """Hann-Poisson window function
        
        Combination of Hann and exponential windows.
        Provides faster sidelobe rolloff than pure Hann window.
        
        Args:
            window_len: Length of the window in samples
            alpha: Shape parameter controlling decay rate
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Hann-Poisson window of specified length
        """
        x = torch.arange(window_len, device=device) / window_len
        return (torch.exp(-alpha * (1 - 2*x).abs()) * 0.5 * (1 - torch.cos(2*torch.pi*x)))

    @staticmethod
    def blackman_harris(window_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Blackman-Harris window function
        
        Four-term cosine window with very low sidelobes.
        Excellent for spectral analysis where dynamic range is important.
        
        Args:
            window_len: Length of the window in samples
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Blackman-Harris window of specified length
        """
        x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
        return (0.35875 - 0.48829 * torch.cos(x) + 0.14128 * torch.cos(2*x) - 0.01168 * torch.cos(3*x))

    @staticmethod
    def flat_top(window_len:int , device: Optional[torch.device] = None) -> torch.Tensor:
        """Flat-top window function
        
        Five-term cosine window optimized for amplitude accuracy.
        Excellent for calibration and amplitude measurements.
        
        Args:
            window_len: Length of the window in samples
            device: Torch device to place the window on
            
        Returns:
            torch.Tensor: Flat-top window of specified length
        """
        x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @staticmethod
    @torch.no_grad()
    def get_window_fn(window_fn: str):
        """Get window function by name
        
        Args:
            window_fn: Name of the window function to retrieve
            
        Returns:
            Callable: The requested window function
        """
        return getattr(WindowFunction, window_fn)

def mclt(x: torch.Tensor, block_width: int, window_fn: str ="hann",
         window_exponent: float = 1., **kwargs) -> torch.Tensor:
    """Modified Complex Lapped Transform
    
    Transforms time-domain signal to frequency domain using overlapping blocks.
    Similar to STFT but with better frequency resolution and phase behavior.
    
    Args:
        x: Input signal tensor
        block_width: Width of processing blocks
        window_fn: Window function to use
        window_exponent: Power to raise window function to
        **kwargs: Additional arguments passed to window function
        
    Returns:
        torch.Tensor: Complex MCLT coefficients
    """
    # Pad signal to handle boundaries
    padding_left = padding_right = block_width // 2
    remainder = x.shape[-1] % (block_width // 2)
    if remainder > 0:
        padding_right += block_width // 2 - remainder

    # Create overlapping blocks
    pad_tuple = (padding_left, padding_right) + (0,0,) * (x.ndim-1)
    x = torch.nn.functional.pad(x, pad_tuple).unfold(-1, block_width, block_width//2)

    # Prepare MCLT parameters
    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    # Apply window function
    window = 1 if window_exponent == 0 else WindowFunction.get_window_fn(window_fn)(
        2*N, device=x.device, **kwargs) ** window_exponent
    
    # Apply pre/post phase shifts
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    # Perform transform
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * (2 * N ** 0.5)

def imclt(x: torch.Tensor, window_fn: str = "hann",
          window_degree: float = 1, **kwargs) -> torch.Tensor:
    """Inverse Modified Complex Lapped Transform
    
    Reconstructs time-domain signal from MCLT coefficients.
    
    Args:
        x: MCLT coefficients
        window_fn: Window function to use
        window_degree: Power to raise window function to
        **kwargs: Additional arguments passed to window function
        
    Returns:
        torch.Tensor: Reconstructed time-domain signal
    """
    # Prepare MCLT parameters
    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    # Apply window function
    window = 1 if window_degree == 0 else WindowFunction.get_window_fn(window_fn)(2*N, device=x.device, **kwargs) ** window_degree

    # Apply pre/post phase shifts
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)

    # Perform inverse transform
    y = (torch.fft.ifft(x / post_shift, norm="backward", n=2*N) / pre_shift) * window

    # Reconstruct time-domain signal from overlapping blocks
    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device, dtype=y.dtype)
    y_even = y[...,  ::2, :].reshape(*y[...,  ::2, :].shape[:-2], -1)
    y_odd  = y[..., 1::2, :].reshape(*y[..., 1::2, :].shape[:-2], -1)
    raw_sample[..., :y_even.shape[-1]] = y_even
    raw_sample[..., N:y_odd.shape[-1] + N] += y_odd

    return raw_sample[..., N:-N] * (2 * N ** 0.5)

def stft(x: torch.Tensor, block_width: int, window_fn: str = "hann", window_degree: float = 1.,
         step: Optional[int] = None, add_channelwise_fft: bool = False, **kwargs) -> torch.Tensor:
    """Short-Time Fourier Transform
    
    Standard STFT implementation for comparison with MCLT.
    
    Args:
        x: Input signal tensor
        block_width: Width of processing blocks
        window_fn: Window function to use
        window_degree: Power to raise window function to
        step: Step size between blocks (default: block_width//2)
        add_channelwise_fft: Whether to perform additional FFT across channels
        **kwargs: Additional arguments passed to window function
        
    Returns:
        torch.Tensor: Complex STFT coefficients
    """
    step = step or block_width//2
    x = x.unfold(-1, block_width, step)

    window = 1 if window_degree == 0 else WindowFunction.get_window_fn(window_fn)(block_width, device=x.device, **kwargs) ** window_degree

    x = torch.fft.rfft(x * window, norm="ortho")
    x = torch.fft.fft(x, norm="ortho", dim=-3) if add_channelwise_fft else x

    return x