import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import wandb  # Optional: for experiment tracking

class TrainingValidator:
    def __init__(self, save_dir, patience=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        
    def update(self, train_loss, val_loss, model, epoch):
        """Update validation stats and handle model saving"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Save best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(model, epoch, val_loss)
        else:
            self.patience_counter += 1
            
        # Plot training progress
        self.plot_losses()
        
        return self.should_stop()
    
    def should_stop(self):
        """Check if early stopping criteria is met"""
        return self.patience_counter >= self.patience
    
    def save_checkpoint(self, model, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.save_dir / f'model_epoch_{epoch}_loss_{val_loss:.4f}.pt')
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'training_progress.png')
        plt.close()
