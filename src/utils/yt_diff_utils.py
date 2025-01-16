"""
This file contains utility functions for the DualDiffusion model framework.

Key functionality:
- MPS (Metal Performance Shaders) initialization and configuration management for Mac
- Audio processing utilities:
  - Loading/saving audio files
  - Audio format conversion and resampling
  - Spectrogram generation and manipulation
- Image processing utilities:
  - Loading/saving images
  - Color space conversions
  - Image transformations and augmentations
- Model utilities:
  - TF32 precision control via context manager
  - Multi-plot visualization helper
  - SafeTensor loading/saving
- File handling:
  - JSON loading/saving
  - File path management
  - Audio metadata handling via mutagen

The utilities in this file provide core functionality used throughout the DualDiffusion 
codebase for data preprocessing, model training, and inference.
"""

import torch
import safetensors.torch as ST
from typing import Optional, Union, Any
import os

def torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Convert string or torch.dtype to torch.dtype"""
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return getattr(torch, dtype)
    else:
        raise ValueError(f"Unsupported dtype type: {dtype} ({type(dtype)})")

def save_safetensors(tensors_dict: dict[str, torch.Tensor], output_path: str,
                     metadata: Optional[dict[str, str]] = None) -> None:
    """Save tensors dictionary to safetensors format"""
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    for key in tensors_dict:
        val = tensors_dict[key]
        if torch.is_tensor(val):
            val = val.detach().resolve_conj().contiguous().cpu()
        else:
            val = torch.tensor(val)
        tensors_dict[key] = val

    ST.save_file(tensors_dict, output_path, metadata=metadata)

def load_safetensors(input_path: str, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    """Load tensors dictionary from safetensors format"""
    return ST.load_file(input_path, device=device)

def normalize(x: torch.Tensor, zero_mean: bool = False) -> torch.Tensor:
    """Normalize tensor"""
    reduction_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)

    if zero_mean:
        x = x - x.mean(dim=reduction_dims, keepdim=True)

    return x / x.square().mean(dim=reduction_dims, keepdim=True).sqrt()
