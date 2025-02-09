from typing import Optional, Union, Literal

import torch

def normalize(x: torch.Tensor, dim: Optional[Union[tuple, list]] = None,
              eps: float = 1e-4) -> torch.Tensor:
    
    norm = torch.linalg.vector_norm(x, dim=dim or list(range(1, x.ndim)),
                                    keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=(norm.numel() / x.numel())**0.5)
    return x / norm.to(x.dtype)

def resample(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.nn.functional.avg_pool2d(x, 2) # should be multiplied by 2 to be magnitude preserving,
    elif mode == 'up':                              # however, pixel norm is applied after downsampling so it doesn't matter
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

def resample_1d(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.lerp(x[..., ::2], x[..., 1::2], 0.5)
    elif mode == 'up':
        return torch.repeat_interleave(x, 2, dim=-1)
    
#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a: torch.Tensor, b: torch.Tensor, t: Union[torch.Tensor, float] = 0.5) -> torch.Tensor:
    return a.lerp(b, t) / ((1 - t) ** 2 + t ** 2) ** 0.5

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a: torch.Tensor, b: torch.Tensor,
           dim: int = 1, t: float = 0.5) -> torch.Tensor:
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = ((Na + Nb) / ((1 - t) ** 2 + t ** 2)) ** 0.5
    wa = C / Na**0.5 * (1 - t)
    wb = C / Nb**0.5 * t
    return torch.cat([wa * a , wb * b], dim=dim)

def mp_cat_interleave(a: torch.Tensor, b: torch.Tensor,
                      dim: int = 1, t: float = 0.5) -> torch.Tensor:
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = ((Na + Nb) / ((1 - t) ** 2 + t ** 2)) ** 0.5
    wa = C / Na**0.5 * (1 - t)
    wb = C / Nb**0.5 * t
    return torch.stack([wa * a , wb * b], dim=dim+1).reshape(
        *a.shape[:dim], a.shape[dim]*2, *a.shape[dim+1:])

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):

    def __init__(self, num_channels: int, bandwidth: float = 1., eps: float = 1e-3) -> None:
        super().__init__()
        
        self.register_buffer('freqs', torch.pi * torch.linspace(0, 1-eps, num_channels).erfinv() * bandwidth)
        self.register_buffer('phases', torch.pi/2 * (torch.arange(num_channels) % 2 == 0).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            y = x.float().ger(self.freqs.float()) + self.phases.float()
        else:
            y = x.float() * self.freqs.float().view(1,-1, 1, 1) + self.phases.float().view(1,-1, 1, 1)
        return (y.cos() * 2**0.5).to(x.dtype)

class MPConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: tuple[int, int], groups: int = 1, stride: int = 1,
                 disable_weight_norm: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)

    #@torch.no_grad() is applied at a higher level
    def normalize_weights(self):
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))