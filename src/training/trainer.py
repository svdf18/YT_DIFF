import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = model.get_device()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate
        )
        self.writer = SummaryWriter(log_dir='runs/vae_training')
        self.best_loss = float('inf')
        
    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Log training metrics
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader, epoch)
                self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
                
                # Model checkpointing
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
            
    def _train_epoch(self, dataloader, epoch):
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Extract spectrogram from batch dictionary
            data = batch['spectrogram'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, original, mu, log_var = self.model(data)
            
            loss, recon_loss, kld_loss = self.model.loss_function(
                recon_batch, original, mu, log_var
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            
            # Log batch metrics
            batch_global_idx = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), batch_global_idx)
            self.writer.add_scalar('train/batch_recon_loss', recon_loss.item(), batch_global_idx)
            self.writer.add_scalar('train/batch_kld_loss', kld_loss.item(), batch_global_idx)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon_loss': f'{recon_loss.item():.4f}',
                'kld_loss': f'{kld_loss.item():.4f}'
            })
            
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kld_loss = total_kld_loss / len(dataloader)
        
        return avg_loss
    
    def _validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract spectrogram from batch dictionary
                data = batch['spectrogram'].to(self.device)
                
                # Forward pass
                recon_batch, original, mu, log_var = self.model(data)
                loss, recon_loss, kld_loss = self.model.loss_function(
                    recon_batch, original, mu, log_var
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kld_loss += kld_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kld_loss = total_kld_loss / len(dataloader)
        
        # Log validation metrics
        self.writer.add_scalar('val/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('val/epoch_recon_loss', avg_recon_loss, epoch)
        self.writer.add_scalar('val/epoch_kld_loss', avg_kld_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,  # Save config for model reconstruction
            'window_params': {      # Save windowing parameters
                'window_length': self.config.window_length if hasattr(self.config, 'window_length') else 80,
                'hop_length': self.config.hop_length if hasattr(self.config, 'hop_length') else 40,
            }
        }
        path = f'checkpoints/vae_epoch_{epoch}_loss_{loss:.4f}.pt'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, path)