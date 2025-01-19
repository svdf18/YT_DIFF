import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_spectrogram(spec_data, title, save_path=None):
    """Plot a single spectrogram"""
    # Get spectrogram and metadata
    spec = spec_data['audio']  # Shape should be [2, 256, T]
    metadata = spec_data['metadata']
    
    print(f"DEBUG: Spectrogram shape: {spec.shape}")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot both channels side by side for stereo
    for i in range(spec.shape[0]):  # Iterate over stereo channels
        plt.subplot(1, 2, i+1)
        
        # Reshape if needed and convert to numpy
        channel_data = spec[i].numpy()  # Should be [256, T]
        if len(channel_data.shape) == 1:
            print(f"WARNING: Reshaping channel data from {channel_data.shape}")
            # Attempt to reshape to 2D
            time_frames = len(channel_data) // 256
            channel_data = channel_data.reshape(256, -1)
            print(f"Reshaped to: {channel_data.shape}")
        
        plt.imshow(
            channel_data,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Channel {i+1}: {metadata['subcategory']}")
        plt.xlabel('Time')
        plt.ylabel('Frequency bin')
    
    plt.suptitle(f"{title}\nCategory: {metadata['category']}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_dataset(processed_dir, output_dir=None, max_per_category=5):
    """Visualize spectrograms from processed dataset"""
    processed_dir = Path(processed_dir)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each category
    for category_dir in processed_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"\nVisualizing {category_dir.name} samples...")
        
        # Get list of PT files
        pt_files = list(category_dir.glob("*.pt"))
        
        # Sample random files if there are too many
        if len(pt_files) > max_per_category:
            pt_files = np.random.choice(pt_files, max_per_category, replace=False)
        
        # Plot each file
        for pt_file in pt_files:
            try:
                spec_data = torch.load(pt_file)
                print(f"\nProcessing {pt_file.name}")
                print(f"Loaded spectrogram shape: {spec_data['audio'].shape}")
                
                if output_dir:
                    save_path = os.path.join(output_dir, f"{pt_file.stem}.png")
                else:
                    save_path = None
                    
                plot_spectrogram(
                    spec_data,
                    f"File: {pt_file.name}",
                    save_path
                )
            except Exception as e:
                print(f"Error processing {pt_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # Visualize spectrograms
    visualize_dataset(
        processed_dir="dataset/processed",
        output_dir="dataset/visualizations",  # Set to None to display instead of save
        max_per_category=5  # Maximum number of samples to visualize per category
    ) 