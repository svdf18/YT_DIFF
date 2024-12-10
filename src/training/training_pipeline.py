import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from src.models.vae import AudioVAE
from src.models.unet_edm import UNetEDM
from src.training.dataset import AudioDataset
from src.configs.model_config import Config, VAEOnlyConfig, EDM2OnlyConfig
from src.validation.validation import TrainingValidator
from src.validation.visualizer import AudioVisualizer

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
    
    # Initialize optimizer
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
    
    # Initialize validator and visualizer
    validator = TrainingValidator(
        save_dir='checkpoints',
        patience=config.training.early_stopping_patience
    )
    visualizer = AudioVisualizer(save_dir='visualizations')
    
    # Setup data loaders
    train_dataset = AudioDataset(
        data_dir=config.training.train_dir,
        config=config,
        window_length=config.training.window_length,
        hop_length=config.training.hop_length
    )
    
    val_dataset = AudioDataset(
        data_dir=config.training.val_dir,
        config=config,
        window_length=config.training.window_length,
        hop_length=config.training.hop_length
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Add gradient clipping
    max_grad_norm = 10.0
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        # Training phase
        model.train()
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            max_grad_norm=max_grad_norm
        )
        
        # Validation phase
        if (epoch + 1) % config.training.validation_interval == 0:
            model.eval()
            val_loss = validate(model, val_loader, config)
            
            # Update validator (handles early stopping and model saving)
            if validator.update(train_loss, val_loss, model, epoch):
                logger.info("Early stopping triggered!")
                break
        
        # Visualization
        if (epoch + 1) % config.training.vis_interval == 0:
            visualize_progress(
                model=model,
                dataloader=val_loader,
                visualizer=visualizer,
                epoch=epoch,
                num_samples=config.training.vis_samples
            )
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}" + 
                   (f", Val Loss: {val_loss:.4f}" if (epoch + 1) % config.training.validation_interval == 0 else ""))

def train_epoch(model, dataloader, optimizer, scheduler, config, max_grad_norm):
    total_loss = 0
    optimizer.zero_grad()
    
    with tqdm(dataloader, desc="Training") as pbar:
        for i, batch in enumerate(pbar):
            data = batch['spectrogram'].to(model.get_device())
            
            if isinstance(model, AudioVAE):
                recon, x, mu, log_var = model(data)
                loss, recons_loss, kld_loss = model.loss_function(recon, x, mu, log_var)
                pbar.set_postfix({'loss': loss.item(), 'recon': recons_loss.item(), 'kld': kld_loss.item()})
            elif isinstance(model, UNetEDM):
                time = torch.linspace(0, 1, steps=data.size(0)).to(model.get_device())
                
                if config.edm2.use_pixel_norm:
                    data_norm = data / (data.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
                    recon = model(data_norm, time)
                    loss = F.mse_loss(recon, data_norm)
                else:
                    recon = model(data, time)
                    loss = F.mse_loss(recon, data)
                
                pbar.set_postfix({'loss': loss.item()})
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, config):
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch['spectrogram'].to(model.get_device())
            
            if isinstance(model, AudioVAE):
                recon, x, mu, log_var = model(data)
                loss, _, _ = model.loss_function(recon, x, mu, log_var)
            elif isinstance(model, UNetEDM):
                time = torch.linspace(0, 1, steps=data.size(0)).to(model.get_device())
                recon = model(data, time)
                loss = F.mse_loss(recon, data)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def visualize_progress(model, dataloader, visualizer, epoch, num_samples=4):
    """Generate and save visualizations of current model performance"""
    model.eval()
    with torch.no_grad():
        # Get a batch of validation data
        batch = next(iter(dataloader))
        data = batch['spectrogram'][:num_samples].to(model.get_device())
        
        if isinstance(model, AudioVAE):
            # For VAE: Show original vs reconstruction
            recon, x, mu, log_var = model(data)
            for i in range(min(num_samples, len(data))):
                visualizer.plot_spectrogram_comparison(
                    x[i].cpu(), 
                    recon[i].cpu(),
                    f"Epoch_{epoch}_Sample_{i+1}_VAE"
                )
            
            # Visualize latent space if we have labels
            if 'metadata' in batch and 'label' in batch['metadata']:
                latents = model.encode(data)[0]  # Get mu vectors
                visualizer.plot_latent_space(
                    latents.cpu(), 
                    batch['metadata']['label'][:num_samples]
                )
        
        elif isinstance(model, UNetEDM):
            # For EDM2: Show denoising progression
            time_steps = torch.linspace(1, 0, steps=4).to(model.get_device())
            for i in range(min(num_samples, len(data))):
                results = []
                for t in time_steps:
                    time_batch = torch.full((1,), t).to(model.get_device())
                    denoised = model(data[i:i+1], time_batch)
                    results.append(denoised[0].cpu())
                
                visualizer.plot_spectrogram_comparison(
                    data[i].cpu(),
                    results[-1],  # Final denoised result
                    f"Epoch_{epoch}_Sample_{i+1}_EDM2"
                )

if __name__ == "__main__":
    config = Config()
    train(config)
