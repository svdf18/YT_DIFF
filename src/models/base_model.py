import torch
import torch.nn as nn
import logging

class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def normalize_input(self, x, eps=1e-8):
        """Pixel-wise normalization"""
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + eps)
    
    def get_device(self):
        """Get the device the model is on"""
        device = next(self.parameters()).device
        return torch.device(device.type)
    
    def count_parameters(self):
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def get_optimal_device():
        """Get the optimal available device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def save_model(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model state"""
        self.load_state_dict(torch.load(path, map_location=self.get_device()))
        self.logger.info(f"Model loaded from {path}")