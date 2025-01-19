import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torchaudio
import json
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
import argparse

def get_metadata_from_summary(wandb_dir: Path, step: int) -> dict:
    """Extract metadata from wandb-summary.json"""
    summary_path = wandb_dir / 'files' / 'wandb-summary.json'
    
    if not summary_path.exists():
        print(f"Warning: No wandb-summary.json found at {summary_path}")
        return {'category': 'Unknown', 'labels': []}
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract audio captions
        audio_info = summary.get('audio', {})
        original_caption = audio_info.get('original', {}).get('caption', '')
        recon_caption = audio_info.get('reconstruction', {}).get('caption', '')
        
        # Extract metrics
        metrics = summary.get('metrics', {})
        
        # Extract file info from metadata
        file_info = summary.get('metadata', {}).get('file_info', {})
        
        return {
            'category': file_info.get('category', 'Unknown'),
            'labels': file_info.get('labels', []),
            'path': file_info.get('path', 'Unknown'),
            'original_caption': original_caption,
            'recon_caption': recon_caption,
            'metrics': {
                'loss': metrics.get('loss', 0.0),
                'recon_loss': metrics.get('recon_loss', 0.0),
                'kl_loss': metrics.get('kl_loss', 0.0),
                'target_snr': metrics.get('target_snr', 0.0)
            }
        }
            
    except Exception as e:
        print(f"Error reading wandb summary: {str(e)}")
        return {'category': 'Unknown', 'labels': []}

def plot_comparison(original: torch.Tensor, 
                   reconstruction: torch.Tensor, 
                   metadata: dict, 
                   step: int,
                   format_processor: SpectrogramFormat,
                   save_path: str = None):
    """Plot original and reconstructed spectrograms side by side"""
    
    # Convert audio to spectrograms using our format processor
    orig_spec = format_processor.raw_to_sample(original.unsqueeze(0))
    recon_spec = format_processor.raw_to_sample(reconstruction.unsqueeze(0))
    
    # Remove batch dimension
    orig_spec = orig_spec.squeeze(0)
    recon_spec = recon_spec.squeeze(0)
    
    print(f"Spectrogram shapes - Original: {orig_spec.shape}, Reconstruction: {recon_spec.shape}")
    
    plt.figure(figsize=(15, 8))
    
    # Plot original spectrograms
    for i in range(orig_spec.shape[0]):
        plt.subplot(2, 2, i+1)
        
        channel_data = orig_spec[i].cpu().numpy()
        plt.imshow(
            channel_data,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Original - Channel {i+1}")
        plt.xlabel('Time')
        plt.ylabel('Frequency bin')
    
    # Plot reconstructed spectrograms
    for i in range(recon_spec.shape[0]):
        plt.subplot(2, 2, i+3)
        
        channel_data = recon_spec[i].cpu().numpy()
        plt.imshow(
            channel_data,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Reconstruction - Channel {i+1}")
        plt.xlabel('Time')
        plt.ylabel('Frequency bin')
    
    # Update the title with complete metadata
    category = metadata.get('category', 'Unknown')
    labels = metadata.get('labels', [])
    metrics = metadata.get('metrics', {})
    
    title = (
        f"Training Step: {step}\n"
        f"Category: {category}\n"
        f"Labels: {', '.join(labels)}\n"
        f"File: {metadata.get('path', 'Unknown')}\n"
        f"Loss: {metrics.get('loss', 0.0):.4f} "
        f"(Recon: {metrics.get('recon_loss', 0.0):.4f}, "
        f"KL: {metrics.get('kl_loss', 0.0):.4f})\n"
        f"Target SNR: {metrics.get('target_snr', 0.0):.1f}"
    )
    
    plt.suptitle(title, y=1.08)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_wandb_artifacts(wandb_dir: str, 
                            output_dir: str = None,
                            max_samples: int = 5,
                            run_path: str = None):
    """Visualize original and reconstructed spectrograms from wandb artifacts
    
    Args:
        wandb_dir: Base wandb directory
        output_dir: Where to save visualizations
        max_samples: Maximum number of samples to visualize
        run_path: Optional specific run path (e.g., 'run-20250117_163931-xftedrrr')
                 If None, uses latest run
    """
    
    # Initialize format processor with same config as training
    format_config = SpectrogramFormatConfig(
        sample_rate=32000,
        sample_raw_channels=2,
        sample_raw_length=320000,
        noise_floor=2e-5,
        
        # Spectrogram settings
        abs_exponent=0.25,
        step_size_ms=8,
        window_duration_ms=64,
        padded_duration_ms=64,
        window_exponent=32,
        window_periodic=True,
        
        # Frequency scale parameters
        num_frequencies=256,
        min_frequency=20,
        max_frequency=20000,
        freq_scale_type="mel",
        freq_scale_norm=None,
        
        # Phase recovery parameters
        num_griffin_lim_iters=200,
        momentum=0.99,
        stereo_coherence=0.67
    )
    
    format_processor = SpectrogramFormat(format_config)
    
    wandb_dir = Path(wandb_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find run directory
    if run_path:
        run_dir = wandb_dir / run_path
        if not run_dir.exists():
            print(f"Specified run directory not found: {run_dir}")
            return
        latest_run = run_dir
    else:
        # Find the latest run directory
        run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and 'run-' in str(d)]
        if not run_dirs:
            print(f"No wandb run directories found in {wandb_dir}")
            return
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    
    print(f"\nProcessing run: {latest_run}")
    
    # Find audio files in the media/audio directory
    audio_dir = latest_run / 'files' / 'media' / 'audio'
    if not audio_dir.exists():
        print(f"No audio directory found at {audio_dir}")
        return
    
    # Group files by step
    audio_files = {}
    for audio_file in audio_dir.glob('*'):
        if 'original' in audio_file.name or 'reconstruction' in audio_file.name:
            step = int(audio_file.stem.split('_')[1])
            if step not in audio_files:
                audio_files[step] = {'original': None, 'reconstruction': None}
            if 'original' in audio_file.name:
                audio_files[step]['original'] = audio_file
            else:
                audio_files[step]['reconstruction'] = audio_file
    
    # Process the latest steps
    steps = sorted(audio_files.keys(), reverse=True)[:max_samples]
    
    for step in steps:
        print(f"\nProcessing step {step}")
        orig_file = audio_files[step]['original']
        recon_file = audio_files[step]['reconstruction']
        
        if orig_file and recon_file:
            try:
                # Load audio files
                orig_audio, sr = torchaudio.load(orig_file)
                recon_audio, _ = torchaudio.load(recon_file)
                
                print(f"Audio shapes - Original: {orig_audio.shape}, Reconstruction: {recon_audio.shape}")
                
                # Get metadata from wandb summary
                metadata = get_metadata_from_summary(latest_run, step)
                
                if output_dir:
                    save_path = output_dir / f"comparison_step_{step}.png"
                else:
                    save_path = None
                
                plot_comparison(
                    orig_audio,
                    recon_audio,
                    metadata,
                    step,
                    format_processor,
                    save_path
                )
            except Exception as e:
                print(f"Error processing step {step}: {str(e)}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training artifacts')
    parser.add_argument('--wandb_dir', default='wandb', help='Directory containing wandb runs')
    parser.add_argument('--output_dir', default='outputs/vae/visualizations', help='Where to save visualizations')
    parser.add_argument('--max_samples', type=int, default=5, help='Maximum number of samples to visualize')
    parser.add_argument('--run_path', default=None, help='Specific run path (e.g., run-20250117_163931-xftedrrr)')
    
    args = parser.parse_args()
    
    visualize_wandb_artifacts(
        wandb_dir=args.wandb_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        run_path=args.run_path
    ) 