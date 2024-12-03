import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from src.models.vae import AudioVAE
from src.models.unet_edm import UNetEDM
from src.training.dataset import AudioDataset
from src.configs.model_config import Config, VAEOnlyConfig, EDM2OnlyConfig

def train(config: Config):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize model
    if isinstance(config, VAEOnlyConfig):
        model = AudioVAE(config)
    elif isinstance(config, EDM2OnlyConfig):
        model = UNetEDM(config)
    else:
        raise ValueError("Unsupported model type")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate, 
        betas=(0.9, 0.99), 
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: min((step + 1) / config.training.lr_warmup_steps, 1.0) ** config.training.lr_decay_exponent
    )
    
    # Setup data
    dataset = AudioDataset(data_dir=config.training.data_dir, config=config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Add gradient clipping
    max_grad_norm = 10.0
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for i, batch in enumerate(pbar):
                # Extract spectrogram from batch dictionary
                data = batch['spectrogram'].to(model.get_device())
                
                # Forward pass (different for VAE and EDM2)
                if isinstance(model, AudioVAE):
                    recon, x, mu, log_var = model(data)
                    loss, recons_loss, kld_loss = model.loss_function(recon, x, mu, log_var)
                elif isinstance(model, UNetEDM):
                    # Generate time embeddings for EDM2
                    time = torch.linspace(0, 1, steps=data.size(0)).to(model.get_device())
                    recon = model(data, time)
                    # Simple reconstruction loss for now
                    loss = torch.nn.functional.mse_loss(recon, data)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % config.training.gradient_accumulation_steps == 0:
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                total_loss += loss.item()
                if isinstance(model, AudioVAE):
                    pbar.set_postfix({'loss': loss.item(), 'recon': recons_loss.item(), 'kld': kld_loss.item()})
                else:
                    pbar.set_postfix({'loss': loss.item()})
                
                # Update learning rate
                scheduler.step()
        
        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"checkpoints/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    config = Config()
    train(config)
