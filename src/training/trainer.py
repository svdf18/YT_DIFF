import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from src.validation.validation import TrainingValidator
from src.validation.visualizer import AudioVisualizer

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config.training
        self.device = model.get_device()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Replace TensorBoard with validator and visualizer
        self.validator = TrainingValidator(save_dir='checkpoints')
        self.visualizer = AudioVisualizer(save_dir='visualizations')
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
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader, epoch)
                
                # Update validator (handles saving best model)
                if self.validator.update(train_loss, val_loss, self.model, epoch):
                    print("Early stopping triggered!")
                    break
                
                # Visualize samples periodically
                if epoch % 5 == 0:  # Every 5 epochs
                    sample_batch = next(iter(val_loader))
                    self._visualize_batch(sample_batch, epoch)
            
    def _train_epoch(self, dataloader, epoch):
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon_loss': f'{recon_loss.item():.4f}',
                'kld_loss': f'{kld_loss.item():.4f}'
            })
            
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def _validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
        return avg_loss
    
    def _visualize_batch(self, batch, epoch):
        """Visualize a batch of data"""
        print(f"\nVisualizing batch for epoch {epoch}")
        self.model.eval()
        with torch.no_grad():
            data = batch['spectrogram'].to(self.device)
            recon_batch, original, mu, log_var = self.model(data)
            
            print(f"Original shape: {original[0].shape}")
            print(f"Reconstruction shape: {recon_batch[0].shape}")
            
            # Plot spectrograms
            self.visualizer.plot_spectrogram_comparison(
                original[0].cpu(), 
                recon_batch[0].cpu(),
                f"Epoch {epoch} Reconstruction"
            )
            print(f"Visualization saved to: {self.visualizer.save_dir}")