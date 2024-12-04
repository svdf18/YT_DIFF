import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path

class AudioVisualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_spectrogram_comparison(self, original, reconstructed, title="Spectrogram Comparison"):
        """Plot original vs reconstructed spectrograms"""
        plt.figure(figsize=(15, 5))
        
        # Original
        plt.subplot(1, 2, 1)
        self.plot_single_spectrogram(original, "Original")
        
        # Reconstructed
        plt.subplot(1, 2, 2)
        self.plot_single_spectrogram(reconstructed, "Reconstructed")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    @staticmethod
    def plot_single_spectrogram(spec, title):
        """Helper function to plot a single spectrogram"""
        if torch.is_tensor(spec):
            spec = spec.cpu().detach().numpy()
        plt.imshow(spec.squeeze(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    
    def plot_latent_space(self, latents, labels=None):
        """Plot 2D visualization of latent space (for VAE)"""
        if latents.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
        else:
            latents_2d = latents
            
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels if labels is not None else None)
        if labels is not None:
            plt.colorbar(scatter)
        plt.title('Latent Space Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(self.save_dir / 'latent_space.png')
        plt.close()
