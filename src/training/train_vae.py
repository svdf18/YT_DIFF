import os
# Set MPS fallback before any other imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
import matplotlib.pyplot as plt
import numpy as np
from accelerate.utils import DistributedDataParallelKwargs

from src.modules.vaes.vae_edm2 import DualDiffusionVAE_EDM2, DualDiffusionVAE_EDM2Config
from src.modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from src.training.module_trainers.vae_trainer import VAETrainer, VAETrainerConfig
from src.dataset.audio_dataset import AudioDataset

# Disable torch compile/inductor for MPS compatibility
torch._dynamo.config.suppress_errors = True
torch._inductor.config.fallback_random = True
torch.backends.cudnn.benchmark = False

# If still having issues, completely disable compilation
torch._dynamo.config.disable = True

@dataclass
class TrainingConfig:
    # Data settings
    data_dir: str = "dataset/processed"
    output_dir: str = "outputs/vae"
    pad_to_length: int = 256
    
    # Optimized batch settings from your working config
    batch_size: int = 128  # Larger batch size that worked
    gradient_accumulation_steps: int = 4  # Keep your accumulation steps
    
    # Reduced overhead settings
    eval_every: int = 500  # Less frequent audio logging
    save_every: int = 1000  # Less frequent checkpoints
    
    # Data loading optimizations that worked for you
    num_workers: int = 8
    pin_memory: bool = False  # Keep this False as it worked better
    persistent_workers: bool = True
    prefetch_factor: int = 3  # Keep your prefetch factor
    
    # Model compilation
    enable_model_compilation: bool = True
    compile_params: dict = field(default_factory=lambda: {
        "fullgraph": True,
        "dynamic": False
    })
    
    # Learning rate schedule from your config
    learning_rate: float = 1e-2
    lr_warmup_steps: int = 5000
    lr_reference_steps: int = 5000
    lr_decay_exponent: float = 1.0
    
    # Optimizer settings from your config
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 0.0
    max_grad_norm: float = 10.0
    
    # MPS specific
    mixed_precision: str = "no"  # MPS compatible
    enable_memory_efficient_attention: bool = True

def validate_sample_rate(config_sample_rate: int) -> None:
    """Validate sample rate consistency across configs"""
    expected_rate = 32000
    if config_sample_rate != expected_rate:
        raise ValueError(f"Sample rate mismatch! Expected {expected_rate}, got {config_sample_rate}")

def main():
    # Initialize accelerator with MPS-compatible settings
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    
    # Create output directory
    output_dir = Path(TrainingConfig.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb with an instance of the config
    if accelerator.is_main_process:
        wandb.init(
            project="dualdiffusion-vae",
            config=vars(TrainingConfig())  # Create an instance and convert to dict
        )
    
    # Set up datasets and dataloaders
    train_dataset = AudioDataset(
        TrainingConfig.data_dir,
        split="train",
        pad_to_length=TrainingConfig.pad_to_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=True,
        num_workers=TrainingConfig.num_workers,
        collate_fn=AudioDataset.collate_fn,
        pin_memory=TrainingConfig.pin_memory,
        persistent_workers=TrainingConfig.persistent_workers
    )
    
    # Initialize VAE model with updated config
    vae_config = DualDiffusionVAE_EDM2Config(
        in_channels=2,  # Stereo audio
        out_channels=2,
        latent_channels=4,
        model_channels=256,
        channel_mult=(1, 2, 4, 8),
        num_layers_per_block=4,
        target_snr=20.0,
        res_balance=0.4,
        attn_balance=0.4,
        dropout=0.0   # Start without dropout
    )
    
    vae = DualDiffusionVAE_EDM2(vae_config)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=TrainingConfig.learning_rate
    )
    
    # Initialize trainer with updated config
    trainer_config = VAETrainerConfig(
        target_snr=20.0,
        res_balance=0.4,
        attn_balance=0.4,
        block_overlap=8,
        block_widths=(8, 16, 32, 64),
        channel_kl_loss_weight=0.1,
        recon_loss_weight=0.1,
        mel_bands=80,
        min_frequency=20,
        max_frequency=20000,
        sample_rate=32000,
        stereo_weight=0.67,
        gradient_clip_val=1.0
    )
    
    # Create spectrogram format config - Updated to match settings
    spectrogram_config = SpectrogramFormatConfig(
        sample_rate=32000,
        step_size_ms=16,
        window_duration_ms=64,
        padded_duration_ms=64,
        num_frequencies=256,
        min_frequency=20,
        max_frequency=20000,
        freq_scale_type="mel"
    )
    
    trainer = VAETrainer(
        config=trainer_config,
        model=vae,
        format_processor=SpectrogramFormat(spectrogram_config),
        accelerator=accelerator
    )
    
    # Prepare for distributed training
    train_dataloader, vae, optimizer = accelerator.prepare(
        train_dataloader, vae, optimizer
    )
    
    # Training loop
    global_step = 0
    while global_step < TrainingConfig.lr_reference_steps:
        for batch in train_dataloader:
            # Training step
            with accelerator.accumulate(vae):
                metrics = trainer.training_step(batch)
                loss = metrics['loss']
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.is_main_process and global_step % 10 == 0:
                wandb.log(metrics, step=global_step)
            
            # Save checkpoint
            if global_step % TrainingConfig.save_every == 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        'model': accelerator.get_state_dict(vae),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step,
                        'config': vae_config
                    }
                    torch.save(
                        checkpoint,
                        output_dir / f'checkpoint_{global_step:06d}.pt'
                    )
            
            # Generate samples
            if global_step % TrainingConfig.eval_every == 0:
                if accelerator.is_main_process:
                    samples = trainer.generate_samples(batch)
                    metadata = samples['metadata']
                    
                    # Structure the log data
                    log_data = {
                        'audio': {
                            'original': wandb.Audio(
                                samples['original'][0].cpu(),
                                sample_rate=32000,
                                caption=f"Original - Category: {metadata['categories']}, Labels: {metadata['labels']}"
                            ),
                            'reconstruction': wandb.Audio(
                                samples['reconstruction'][0].cpu(),
                                sample_rate=32000,
                                caption=f"Reconstruction of {metadata['paths']}\nCategory: {metadata['categories']}, Labels: {metadata['labels']}"
                            )
                        },
                        'metrics': {
                            'loss': loss.item(),
                            'recon_loss': metrics['recon_loss'],
                            'kl_loss': metrics['kl_loss'],
                            'target_snr': samples['target_snr']
                        },
                        'metadata': {
                            'file_info': {
                                'path': metadata['paths'],
                                'category': metadata['categories'],
                                'labels': metadata['labels']
                            }
                        }
                    }
                    
                    wandb.log(log_data, step=global_step)
            
            global_step += 1
            if global_step >= TrainingConfig.lr_reference_steps:
                break
    
    # Final save
    if accelerator.is_main_process:
        checkpoint = {
            'model': accelerator.get_state_dict(vae),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
            'config': vae_config
        }
        torch.save(
            checkpoint,
            output_dir / 'checkpoint_final.pt'
        )

    # Validate sample rates
    validate_sample_rate(32000)
    validate_sample_rate(spectrogram_config.sample_rate)

if __name__ == "__main__":
    main()
