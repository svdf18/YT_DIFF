# src/pipelines/yt_diff_pipeline.py

import utils.config as config

import os
import importlib
from dataclasses import dataclass
from typing import Optional, Union, Any
from datetime import datetime
import multiprocessing.managers

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from models.base_model import BaseModel
from models.unet_edm import UNetEDM
from utils.dual_diffusion_utils import normalize, load_safetensors, torch_dtype



@dataclass
class SampleParams:
    seed: Optional[int] = None
    num_steps: int = 100
    batch_size: int = 1
    length: Optional[int] = None
    seamless_loop: bool = False
    cfg_scale: float = 1.5
    sigma_max: Optional[float] = None
    sigma_min: Optional[float] = None
    sigma_data: Optional[float] = None
    rho: float = 7.
    schedule: Optional[str] = "edm2"
    tags: Optional[dict] = None  # Changed from prompt to tags
    use_heun: bool = True
    input_perturbation: float = 1.
    conditioning_perturbation: float = 0.
    num_fgla_iters: int = 250
    input_audio: Optional[Union[str, torch.Tensor]] = None
    input_audio_pre_encoded: bool = False

    def sanitize(self) -> "SampleParams":
        self.seed = int(self.seed) if self.seed is not None else None
        self.length = int(self.length) if self.length is not None else None
        self.num_steps = int(self.num_steps)
        self.batch_size = int(self.batch_size)
        self.num_fgla_iters = int(self.num_fgla_iters)
        return self

    def get_metadata(self) -> dict[str, Any]:
        metadata = self.__dict__.copy()
        if metadata["input_audio"] is not None and (not isinstance(metadata["input_audio"], str)):
            metadata["input_audio"] = True
        metadata["timestamp"] = datetime.now().strftime(r"%m/%d/%Y %I:%M:%S %p")
        return {str(key): str(value) for key, value in metadata.items()}

@dataclass
class SampleOutput:
    raw_sample: torch.Tensor
    spectrogram: torch.Tensor
    params: SampleParams
    debug_info: dict[str, Any]
    latents: Optional[torch.Tensor] = None

@dataclass
class ModuleInventory:
    name: str
    checkpoints: list[str]
    emas: dict[str, list[str]]

class YTDiffPipeline(torch.nn.Module):
    def __init__(self, pipeline_modules: dict[str, BaseModel]) -> None:
        super().__init__()
        
        for module_name, module in pipeline_modules.items():
            if not isinstance(module, BaseModel):
                raise ValueError(f"Module '{module_name}' must be an instance of BaseModel")
            
            self.add_module(module_name, module)

    @staticmethod
    def from_pretrained()  # For loading models
    
    def save_pretrained()  # For saving models
    
    def get_tag_embeddings()  # For handling music tag conditioning
    
    @torch.inference_mode()
    def __call__()  # The main generation pipeline