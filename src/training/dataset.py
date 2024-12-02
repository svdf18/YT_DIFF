import torch
import torchaudio
import glob
import os
import numpy as np
from pathlib import Path

# 3. Create dataset loader (src/training/dataset.py)
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config, window_length=80, hop_length=40):
        self.data_dir = Path(data_dir)
        self.config = config
        self.window_length = window_length
        self.hop_length = hop_length
        self.files = self.get_file_list()
        
        if len(self.files) == 0:
            raise ValueError(f"No valid files found in {self.data_dir}")
        
        # Pre-calculate number of windows for each file
        self.file_windows = []
        for file_path in self.files:
            data = torch.load(file_path)
            spec_length = data['spectrogram'].shape[2]
            num_windows = max(1, (spec_length - window_length) // hop_length + 1)
            self.file_windows.append(num_windows)
            
        self.cumulative_windows = np.cumsum([0] + self.file_windows)
    
    def get_file_list(self):
        # Get all spectrogram files (both .npy and .pt formats)
        spect_files = []
        for ext in ['*.npy', '*.pt']:
            spect_files.extend(self.data_dir.glob(ext))
        
        # If no spectrograms found, try audio files
        if not spect_files:
            audio_files = []
            for ext in ['*.wav', '*.mp3']:
                audio_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            return sorted(audio_files)
            
        return sorted(spect_files)
    
    def __len__(self):
        return self.cumulative_windows[-1]

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_windows[1:], idx, side='right')
        window_idx = idx - self.cumulative_windows[file_idx]
        
        # Load the file
        data = torch.load(self.files[file_idx])
        spectrogram = data['spectrogram']
        
        # Calculate start position for this window
        start_pos = window_idx * self.hop_length
        window = spectrogram[:, :, start_pos:start_pos + self.window_length]
        
        # If this is the last window, it might be shorter - pad if necessary
        if window.shape[2] < self.window_length:
            pad_size = self.window_length - window.shape[2]
            window = torch.nn.functional.pad(window, (0, pad_size))
        
        return {
            'spectrogram': window,
            'file_idx': file_idx,
            'window_idx': window_idx,
            'metadata': data.get('metadata', {}),
            'original_path': data['original_path']
        }
