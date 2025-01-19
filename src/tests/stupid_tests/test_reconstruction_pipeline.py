import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from src.utils.config import load_json

def plot_comparison(original_spec: torch.Tensor, 
                   reconstructed_spec: torch.Tensor, 
                   save_path: Optional[str] = None):
    """Plot original vs reconstructed spectrograms side by side"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original spectrograms (left channel, right channel)
    for i in range(2):
        axes[0][i].imshow(
            original_spec[i].numpy(),
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma'
        )
        axes[0][i].set_title(f'Original - Channel {i+1}')
        axes[0][i].set_xlabel('Time')
        axes[0][i].set_ylabel('Frequency bin')
    
    # Plot reconstructed spectrograms
    for i in range(2):
        axes[1][i].imshow(
            reconstructed_spec[i].numpy(),
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma'
        )
        axes[1][i].set_title(f'Reconstructed - Channel {i+1}')
        axes[1][i].set_xlabel('Time')
        axes[1][i].set_ylabel('Frequency bin')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Convert mel scale to Hz"""
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """Convert Hz to mel scale"""
    return 2595.0 * torch.log10(1.0 + (freq / 700.0))

def create_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Create a mel filterbank matrix"""
    # Calculate mel points
    min_mel = hz_to_mel(torch.tensor(20.0))  # Min frequency 20 Hz
    max_mel = hz_to_mel(torch.tensor(sample_rate / 2))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Create filterbank
    bins = torch.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    fbank = torch.zeros((n_mels, len(bins)))
    
    # Create triangular filters
    for i in range(n_mels):
        lower = hz_points[i]
        peak = hz_points[i + 1]
        upper = hz_points[i + 2]
        
        # Lower slope
        lower_mask = (bins >= lower) & (bins <= peak)
        fbank[i, lower_mask] = (bins[lower_mask] - lower) / (peak - lower)
        
        # Upper slope
        upper_mask = (bins >= peak) & (bins <= upper)
        fbank[i, upper_mask] = (upper - bins[upper_mask]) / (upper - peak)
    
    return fbank

def hann_power_window(window_length: int, periodic: bool = True, exponent: float = 1.0) -> torch.Tensor:
    """Create a powered Hann window"""
    window = torch.hann_window(window_length, periodic=periodic)
    return window ** exponent

def test_reconstruction(pt_file_path: str, output_dir: str):
    """Test reconstruction pipeline on a single .pt file"""
    print(f"\nTesting reconstruction for: {pt_file_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the .pt file
    spec_data = torch.load(pt_file_path)
    original_spec = spec_data['audio']
    metadata = spec_data['metadata']
    
    print(f"Original spectrogram shape: {original_spec.shape}")
    
    try:
        # Parameters for reconstruction
        sample_rate = 32000
        n_fft = 2048
        hop_length = int(8 * sample_rate / 1000)  # 8ms step size
        win_length = int(64 * sample_rate / 1000)  # 64ms window
        window_exponent = 32.0
        
        # Create powered window
        window = hann_power_window(win_length, periodic=True, exponent=window_exponent)
        
        # Initialize Griffin-Lim
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=lambda size: window,
            power=1.0,
            n_iter=128,  # Increased iterations
            momentum=0.99,
            length=None,
            rand_init=False  # Changed to false for more stable init
        )
        
        # Process each channel separately
        reconstructed_channels = []
        for channel in range(original_spec.shape[0]):
            # Prepare spectrogram for Griffin-Lim
            spec = original_spec[channel]  # [F, T]
            
            # Unscale from power law (use original config value)
            spec = spec ** (1 / 0.25)  # Inverse of 0.25 power
            
            # Pad to full frequency resolution
            padded_spec = torch.zeros(n_fft // 2 + 1, spec.shape[1])
            padded_spec[:spec.shape[0]] = spec
            
            # Reconstruct audio
            audio = griffin_lim(padded_spec)
            reconstructed_channels.append(audio)
        
        # Stack channels
        reconstructed_audio = torch.stack(reconstructed_channels)
        print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
        
        # Save audio
        base_name = Path(pt_file_path).stem
        torchaudio.save(
            str(output_dir / f"{base_name}_reconstructed.wav"),
            reconstructed_audio,
            sample_rate
        )
        
        # Convert back to spectrogram for comparison
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=lambda size: window,  # Use same powered window
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True
        )
        
        reconstructed_spec = []
        for channel in range(reconstructed_audio.shape[0]):
            spec = spectrogram(reconstructed_audio[channel])
            # Take magnitude
            spec = spec.abs()
            # Cut to original frequency bins
            spec = spec[:original_spec.shape[1]]
            # Apply same power as original
            spec = spec ** 0.25
            reconstructed_spec.append(spec)
        
        reconstructed_spec = torch.stack(reconstructed_spec)
        print(f"Reconstructed spectrogram shape: {reconstructed_spec.shape}")
        
        # Save comparison plot
        plot_comparison(
            original_spec,
            reconstructed_spec,
            save_path=str(output_dir / f"{base_name}_comparison.png")
        )
        
        print(f"Results saved to: {output_dir}")
        
        return {
            'original_spec': original_spec,
            'reconstructed_spec': reconstructed_spec,
            'reconstructed_audio': reconstructed_audio,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
        print(f"Last known shapes:")
        print(f"Original spec: {original_spec.shape}")
        if 'reconstructed_audio' in locals():
            print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
        if 'reconstructed_spec' in locals():
            print(f"Reconstructed spec shape: {reconstructed_spec.shape}")
        
        return {
            'original_spec': original_spec,
            'metadata': metadata,
            'error': str(e)
        }

def test_dataset_samples(processed_dir: str, 
                        output_dir: str,
                        max_per_category: int = 2):
    """Test reconstruction on multiple samples from the processed dataset"""
    processed_dir = Path(processed_dir)
    
    # Process each category
    for category_dir in processed_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"\nTesting {category_dir.name} samples...")
        
        # Get list of PT files
        pt_files = list(category_dir.glob("*.pt"))
        
        # Sample random files if there are too many
        if len(pt_files) > max_per_category:
            import numpy as np
            pt_files = np.random.choice(pt_files, max_per_category, replace=False)
        
        # Test each file
        for pt_file in pt_files:
            try:
                test_reconstruction(
                    str(pt_file),
                    output_dir=str(Path(output_dir) / category_dir.name)
                )
            except Exception as e:
                print(f"Error processing {pt_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # Test reconstruction pipeline on dataset samples
    test_dataset_samples(
        processed_dir="dataset/processed",
        output_dir="dataset/reconstruction_tests",
        max_per_category=2
    )
