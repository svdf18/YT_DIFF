import torch
import torchaudio
import glob
import os

# 3. Create dataset loader (src/training/dataset.py)
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.files = self.get_file_list()
        
    def get_file_list(self):
        # Get all audio files in the directory
        audio_files = []
        for ext in ['*.wav', '*.mp3']:
            audio_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
        return sorted(audio_files)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform
