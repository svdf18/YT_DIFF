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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Setup data
    dataset = AudioDataset(data_dir=config.data_dir, config=config)  # Pass both data_dir and config
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Extract spectrogram from batch dictionary
                data = batch['spectrogram'].to(model.get_device())
                
                # Forward pass
                recon, orig, mu, logvar = model(data)
                loss, recon_loss, kld = model.loss_function(recon, orig, mu, logvar)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
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
