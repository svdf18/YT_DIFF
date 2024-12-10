# 2. Create dataset preprocessing (src/utils/preprocess.py)
import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from src.modules.formats.spectogram import SpectrogramFormat
import argparse
from src.configs.base_config import AudioConfig

class AudioPreprocessor:
    def __init__(self, config):
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.format = SpectrogramFormat(config)

        # Validate directories
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing audio files from: {self.input_dir}")
        print(f"Saving spectrograms to: {self.output_dir}")
        
    def process_file(self, audio_path, window_length=80, hop_length=40):
        """Process a single audio file to spectrogram with sliding windows
        
        Args:
            audio_path: Path to audio file
            window_length: Number of frames per window (default 80)
            hop_length: Number of frames to slide between windows (default 40)
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.format.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.format.sample_rate)
            audio = resampler(audio)
        
        # Convert to spectrogram
        spec = self.format.audio_to_spectrogram(audio)
        
        # Store original spectrogram length
        original_length = spec.shape[2]
        
        # Create output path
        rel_path = Path(audio_path).relative_to(self.input_dir)
        output_path = self.output_dir / rel_path.with_suffix('.pt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed data with metadata
        torch.save({
            'spectrogram': spec,
            'original_path': str(audio_path),
            'sample_rate': self.format.sample_rate,
            'window_length': window_length,
            'hop_length': hop_length,
            'original_length': original_length,
            # Add any additional metadata/tags here
            'metadata': {
                'filename': audio_path.name,
                # You could add more metadata like:
                # 'tags': ['bass', 'reese', 'C#'],
                # 'category': 'bass',
                # 'key': 'C#',
                # etc.
            }
        }, output_path)
        
        return output_path
    
    def process_directory(self):
        """Process all audio files in input directory"""
        audio_files = []
        for ext in ['.wav', '.mp3']:
            found_files = list(self.input_dir.rglob(f'*{ext}'))
            print(f"Found {len(found_files)} {ext} files")
            audio_files.extend(found_files)
        
        if not audio_files:
            raise ValueError(f"No audio files found in {self.input_dir}")
        
        print(f"Processing {len(audio_files)} total files...")
        
        processed_files = []
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            try:
                output_path = self.process_file(audio_path)
                processed_files.append(output_path)
                print(f"Processed {audio_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                
        return processed_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=False)
    parser.add_argument('--split', type=str, choices=['training', 'validation'], default=None)
    args = parser.parse_args()

    # Create config with the directories
    config = AudioConfig()
    
    # Update paths based on split
    if args.split:
        config.input_dir = f"data/{args.split}/raw"
        config.output_dir = f"data/{args.split}/processed"
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("--input_dir and --output_dir are required when --split is not used")
        config.input_dir = args.input_dir
        config.output_dir = args.output_dir

    # Create preprocessor and process files
    preprocessor = AudioPreprocessor(config)
    processed_files = preprocessor.process_directory()
    print(f"Processed {len(processed_files)} files")

if __name__ == "__main__":
    main()
