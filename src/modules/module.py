from utils import config

import os
import inspect
from typing import Optional, Type
from abc import ABC
from dataclasses import dataclass

import torch

from src.utils.yt_diff_utils import load_safetensors, save_safetensors, torch_dtype

@dataclass
class DualDiffusionModuleConfig(ABC):
    """Base configuration class for all DualDiffusion modules"""
    last_global_step: int = 0

class DualDiffusionModule(torch.nn.Module, ABC):
    """Base module class for all DualDiffusion modules
    
    Provides common functionality for model loading/saving, device/dtype handling,
    weight management, and compilation support.
    """
    
    config_class: Optional[Type[DualDiffusionModuleConfig]] = None
    module_name: Optional[str] = None
    has_trainable_parameters: bool = True
    supports_half_precision: bool = True
    supports_compile: bool = True

    def __init__(self):
        super().__init__()
        
        self.dtype = torch.get_default_dtype()
        self.device = torch.device("cpu")
        self.memory_format = torch.contiguous_format
        self.module_path = None
        
    @classmethod
    @torch.no_grad()
    def from_pretrained(cls: Type["DualDiffusionModule"],
                        module_path: str,
                        subfolder: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None,
                        load_config_only: bool = False
                        ) -> "DualDiffusionModule":
        """Load a pretrained model from disk
        
        Args:
            module_path: Path to the module directory
            subfolder: Optional subfolder within module_path
            torch_dtype: Optional dtype to load the model in
            device: Optional device to load the model to
            load_config_only: If True, only load the config without weights
        """
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        
        config_class = cls.config_class or inspect.signature(cls.__init__).parameters["config"].annotation
        module_name = os.path.basename(module_path)
        module_config = config_class(**config.load_json(os.path.join(module_path, f"{module_name}.json")))

        module = cls(module_config).requires_grad_(False).train(False)
        
        if (not load_config_only) and cls.has_trainable_parameters:
            module.load_state_dict(load_safetensors(os.path.join(module_path, f"{module_name}.safetensors")))
        
        module.module_path = module_path
        return module.to(dtype=torch_dtype, device=device)
    
    @torch.no_grad()
    def save_pretrained(self, module_path: str, subfolder: Optional[str] = None) -> None:
        """Save the model to disk
        
        Args:
            module_path: Path to save the module to
            subfolder: Optional subfolder within module_path
        """
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        os.makedirs(module_path, exist_ok=True)
        
        module_name = os.path.basename(module_path)

        config.save_json(self.config.__dict__, os.path.join(module_path, f"{module_name}.json"))
        if type(self).has_trainable_parameters:
            save_safetensors(self.state_dict(), os.path.join(module_path, f"{module_name}.safetensors"))

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
           memory_format: Optional[torch.memory_format] = None,**kwargs) -> "DualDiffusionModule":
        """Move module to specified device/dtype/memory_format"""
        
        if dtype is not None:
            if type(self).supports_half_precision == True:
                dtype = torch_dtype(dtype)
            else:
                dtype = torch.float32

        super().to(device=device, dtype=dtype, memory_format=memory_format, **kwargs)

        self.dtype = dtype or self.dtype
        self.device = device or self.device
        self.memory_format = memory_format or self.memory_format

        return self
    
    def half(self) -> "DualDiffusionModule":
        """Convert module to half precision (bfloat16)"""
        if type(self).supports_half_precision == True:
            self.dtype = torch.bfloat16
            return self.to(dtype=torch.bfloat16)
        else:
            return self
    
    def compile(self, **kwargs) -> None:
        """Compile the module using torch.compile"""
        if type(self).supports_compile == True:
            self.forward = torch.compile(self.forward, **kwargs)
            if hasattr(self, "normalize_weights") and self.training == True:
                self.normalize_weights = torch.compile(self.normalize_weights, **kwargs)

    @torch.no_grad()
    def load_ema(self, ema_path: str) -> None:
        """Load EMA weights from disk"""
        self.load_state_dict(load_safetensors(ema_path))
        self.normalize_weights()

    @torch.no_grad()
    def blend_weights(self, other: "DualDiffusionModule", t: float = 0.5) -> None:
        """Blend weights with another module
        
        Args:
            other: Other module to blend with
            t: Blending factor (0=self, 1=other)
        """
        for ((param_name, param), (other_param_name, other_param)) in zip(self.named_parameters(), other.named_parameters()):
            if param.data.shape != other_param.data.shape:
                raise ValueError(f"Cannot blend parameters with different shapes: {param_name} {param.data.shape} != {other_param_name} {other_param.data.shape}")
            param.data.lerp_(other_param.data, t)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Normalize weights of all submodules"""
        if type(self).has_trainable_parameters == False: return
        for module in self.modules():
            if hasattr(module, "normalize_weights") and module != self:
                module.normalize_weights()
