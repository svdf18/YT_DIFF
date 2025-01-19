import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import json
import torch
import torchaudio
import librosa
from pathlib import Path
from tqdm import tqdm
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig

class SimpleAudioProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create format processor config with settings matching training config
        self.format_config = SpectrogramFormatConfig(
            # Base settings from dataset.json
            sample_rate=self.config['sample_rate'],        # 32000
            sample_raw_channels=self.config['num_channels'],  # 2
            sample_raw_length=self.config['max_sample_length'],  # 320000
            noise_floor=2e-5,
            
            # Spectrogram settings - matched with training config
            abs_exponent=0.25,
            step_size_ms=16,           # Controls hop_length
            window_duration_ms=64,     # Controls window size
            padded_duration_ms=64,     # Same as window_duration for no padding
            window_exponent=32,
            window_periodic=True,
            
            # Frequency scale parameters
            num_frequencies=256,       # Changed from 128 to 256
            min_frequency=20,
            max_frequency=16000,
            freq_scale_type="mel",     # Using mel scale
            freq_scale_norm=None,      # No additional normalization
            
            # Phase recovery parameters
            num_griffin_lim_iters=200,
            momentum=0.99,
            stereo_coherence=0.67
        )
        
        # Initialize format processor
        self.format_processor = SpectrogramFormat(self.format_config)
        
        # Create category mapping from config
        self.subcategory_to_category = {}
        for category, subcats in self.config['category_mapping'].items():
            for subcat in subcats:
                self.subcategory_to_category[subcat] = category
        
        self.categories = list(self.config['clap_embedding_labels'].keys())
        
    def process_audio_file(self, audio_path, subcategory):
        """Process a single audio file with category label"""
        try:
            print(f"\nProcessing file: {audio_path}")
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            print(f"Loaded waveform shape: {waveform.shape}, sr: {sr}")
            
            # Resample if necessary
            if sr != self.format_config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.format_config.sample_rate)
                waveform = resampler(waveform)
                print(f"Resampled waveform shape: {waveform.shape}")
            
            # Convert to stereo if mono
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
                print(f"Converted to stereo shape: {waveform.shape}")
            
            # Ensure waveform is the right shape before processing
            # Should be [channels, samples] -> [batch, channels, samples]
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            print(f"Waveform shape after adding batch dim: {waveform.shape}")
            
            # Process through format processor
            print("Converting to spectrogram...")
            processed = self.format_processor.raw_to_sample(waveform)
            print(f"Processed spectrogram shape: {processed.shape}")
            
            # Remove batch dimension for storage
            processed = processed.squeeze(0)
            print(f"Final spectrogram shape: {processed.shape}")
            
            # Get main category
            main_category = self.subcategory_to_category[subcategory]
            
            # Create metadata
            metadata = {
                'category': main_category,
                'subcategory': subcategory,
                'labels': self.config['clap_embedding_labels'][subcategory],
                'format': 'spectrogram'
            }
            
            return processed, metadata
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_dataset(self, input_dir, output_dir):
        """Process all audio files with flat category structure"""
        # Clear and recreate output directory
        if os.path.exists(output_dir):
            print(f"Clearing existing output directory: {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of supported formats
        formats = self.config['dataset_formats']
        
        # Process each category (snare, kick, 808, sine)
        for category in self.categories:
            category_dir = os.path.join(input_dir, category)
            output_category_dir = os.path.join(output_dir, category)
            
            if not os.path.exists(category_dir):
                print(f"Warning: {category_dir} does not exist, skipping")
                continue
                
            os.makedirs(output_category_dir, exist_ok=True)
            
            files = []
            for fmt in formats:
                files.extend(Path(category_dir).glob(f"*{fmt}"))
            
            print(f"\nProcessing {category} samples...")
            for audio_path in tqdm(files):
                result = self.process_audio_file(audio_path, category)
                if result is not None:
                    waveform, metadata = result
                    # Save as PT file with metadata
                    output_path = os.path.join(
                        output_category_dir, 
                        f"{audio_path.stem}.pt"
                    )
                    torch.save({
                        'audio': waveform,
                        'metadata': metadata
                    }, output_path)

if __name__ == "__main__":
    print("Starting audio processing...")
    
    # Initialize processor with config
    processor = SimpleAudioProcessor("config/dataset/dataset.json")
    
    # Set input and output directories
    input_dir = "dataset/raw_audio"  # Your input audio files directory
    output_dir = "dataset/processed"  # Where to save processed files
    
    # Process the dataset
    processor.process_dataset(input_dir, output_dir)
