import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from src.models.vae import AudioVAE
from src.training.dataset import AudioDataset
from src.configs.base_config import TrainingConfig

def train(config: TrainingConfig):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize model
    model = AudioVAE(config)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(0.9, 0.99), 
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: min((step + 1) / config.lr_warmup_steps, 1.0) ** config.lr_decay_exponent
    )
    
    # Setup data
    dataset = AudioDataset(data_dir=config.data_dir, config=config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,  # Increase number of workers
        pin_memory=True
    )
    
    # Add gradient clipping
    max_grad_norm = 10.0
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for i, batch in enumerate(pbar):
                # Extract spectrogram from batch dictionary
                data = batch['spectrogram'].to(model.get_device())
                
                # Forward pass
                recon, orig, mu, logvar = model(data)
                loss, recon_loss, kld = model.loss_function(recon, orig, mu, logvar)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Update learning rate
                scheduler.step()

                print(f"Input range: [{data.min():.2f}, {data.max():.2f}]")
                print(f"Reconstruction range: [{recon.min():.2f}, {recon.max():.2f}]")
        
        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Recon: {recon_loss:.4f}, KLD: {kld:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"checkpoints/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
