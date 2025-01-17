import os
# Set MPS fallback before any other imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb

from src.modules.vaes.vae_edm2 import DualDiffusionVAE_EDM2, DualDiffusionVAE_EDM2Config
from src.modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from src.training.module_trainers.vae_trainer import VAETrainer, VAETrainerConfig
from src.dataset.audio_dataset import AudioDataset

@dataclass
class TrainingConfig:
    # Data settings
    data_dir: str = "dataset/processed"
    output_dir: str = "outputs/vae"
    pad_to_length: int = 256
    
    # Training settings
    batch_size: int = 16
    num_workers: int = 4
    max_steps: int = 100_000
    save_every: int = 1000
    eval_every: int = 100
    
    # Model settings
    model_channels: int = 96
    latent_channels: int = 4
    channel_mult: Tuple[int, ...] = (1, 2, 3, 5)
    num_layers_per_block: int = 3
    
    # Optimizer settings
    learning_rate: float = 1e-4
    
    # Loss settings
    block_overlap: int = 8
    block_widths: Tuple[int, ...] = (8, 16, 32, 64)
    channel_kl_loss_weight: float = 0.1
    recon_loss_weight: float = 0.1

def main():
    # Initialize accelerator with appropriate mixed precision setting
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb"
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
        collate_fn=AudioDataset.collate_fn
    )
    
    # Initialize VAE model
    vae_config = DualDiffusionVAE_EDM2Config(
        in_channels=2,  # Stereo audio
        out_channels=2,
        latent_channels=TrainingConfig.latent_channels,
        model_channels=TrainingConfig.model_channels,
        channel_mult=TrainingConfig.channel_mult,
        num_layers_per_block=TrainingConfig.num_layers_per_block,
        label_dim=0,  # No class conditioning
        dropout=0.0   # Start without dropout
    )
    
    vae = DualDiffusionVAE_EDM2(vae_config)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=TrainingConfig.learning_rate
    )
    
    # Initialize trainer
    trainer_config = VAETrainerConfig(
        block_overlap=TrainingConfig.block_overlap,
        block_widths=TrainingConfig.block_widths,
        channel_kl_loss_weight=TrainingConfig.channel_kl_loss_weight,
        recon_loss_weight=TrainingConfig.recon_loss_weight
    )
    
    # Create spectrogram format config
    spectrogram_config = SpectrogramFormatConfig(
        sample_rate=32000,          # Sample rate stays the same
        step_size_ms=16,           # This controls hop_length
        window_duration_ms=64,     # This controls window size
        padded_duration_ms=64,     # Same as window_duration_ms for no padding
        num_frequencies=256,       # Changed from 128 to 256 to match VAE output
        min_frequency=20,         
        max_frequency=16000,      
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
    while global_step < TrainingConfig.max_steps:
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
                    wandb.log({
                        'original': wandb.Audio(
                            samples['original'][0].cpu(),
                            sample_rate=32000
                        ),
                        'reconstruction': wandb.Audio(
                            samples['reconstruction'][0].cpu(),
                            sample_rate=32000
                        )
                    }, step=global_step)
            
            global_step += 1
            if global_step >= TrainingConfig.max_steps:
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

if __name__ == "__main__":
    main()
