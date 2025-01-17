from pathlib import Path
from typing import Optional, Union, List, Tuple

import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """Dataset for loading preprocessed audio spectrograms from .pt files"""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = "train",
                 max_files: Optional[int] = None,
                 pad_to_length: Optional[int] = None):
        """Initialize dataset
        
        Args:
            data_dir: Directory containing .pt files
            split: Dataset split ('train' or 'valid')
            max_files: Maximum number of files to load (for debugging)
            pad_to_length: Length to pad/crop spectrograms to (None = keep original)
        """
        self.data_dir = Path(data_dir)
        self.pad_to_length = pad_to_length
        
        # Get all .pt files from category subdirectories
        self.file_paths = []
        for category_dir in self.data_dir.glob("*"):
            if category_dir.is_dir():
                self.file_paths.extend(sorted(category_dir.glob("*.pt")))
        
        if max_files:
            self.file_paths = self.file_paths[:max_files]
            
        if len(self.file_paths) == 0:
            raise ValueError(f"No .pt files found in {self.data_dir}")
            
        # Load first file to get data shape
        sample = torch.load(self.file_paths[0])
        self.audio_shape = sample['audio'].shape
        
        print(f"Found {len(self.file_paths)} files in {self.data_dir}")
        print(f"Base audio shape: {self.audio_shape}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _pad_or_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or crop spectrogram to target length"""
        # Remove batch dimension if present
        if audio.dim() == 4:
            audio = audio.squeeze(0)
            
        current_length = audio.shape[-1]
        
        if self.pad_to_length is None or current_length == self.pad_to_length:
            return audio
            
        if current_length < self.pad_to_length:
            # Pad
            pad_amount = self.pad_to_length - current_length
            return torch.nn.functional.pad(audio, (0, pad_amount))
        else:
            # Crop from center
            start = (current_length - self.pad_to_length) // 2
            return audio[..., start:start + self.pad_to_length]

    def __getitem__(self, idx: int) -> dict:
        """Load a single spectrogram file
        
        Returns:
            dict containing:
                'audio': torch.Tensor [2, 256, L] - Stereo spectrogram
                'path': str - Path to source file
                'category': str - Audio category
                'labels': List[str] - Audio labels
        """
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        
        # Process audio
        audio = self._pad_or_crop(data['audio'])
        
        return {
            'audio': audio,
            'path': str(file_path),
            'category': data.get('category', 'unknown'),
            'labels': data.get('labels', [])
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Collate batch of samples
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            dict containing:
                'audio': torch.Tensor [B, 2, 256, L]
                'paths': List[str]
                'categories': List[str]
                'labels': List[List[str]]
        """
        return {
            'audio': torch.stack([item['audio'] for item in batch]),
            'paths': [item['path'] for item in batch],
            'categories': [item['category'] for item in batch],
            'labels': [item['labels'] for item in batch]
        }
