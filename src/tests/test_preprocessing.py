# tests/test_preprocessing.py
import pytest
import torch
import torchaudio
import os
from pathlib import Path
from src.utils.preprocess import AudioPreprocessor
from src.configs.base_config import AudioConfig
from dataclasses import dataclass

@dataclass
class PreprocessingTestConfig(AudioConfig):
    """Extends AudioConfig with preprocessing-specific paths"""
    def __init__(self):
        super().__init__()
        self.input_dir = "src/tests/test_data/audio"
        self.output_dir = "src/tests/test_data/processed"
        self.n_mels = 80  # Use a more typical value

@pytest.fixture(scope="session")
def test_audio_file(tmp_path_factory):
    """Create a test audio file"""
    # Create test directory if it doesn't exist
    test_dir = Path("src/tests/test_data/audio")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple sine wave
    sample_rate = 44100
    duration = 1  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    wave = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave
    
    # Save as wav file
    file_path = test_dir / "test_sine.wav"
    torchaudio.save(file_path, wave.unsqueeze(0), sample_rate)
    
    return file_path

@pytest.fixture
def test_config():
    return PreprocessingTestConfig()

@pytest.fixture
def preprocessor(test_config):
    return AudioPreprocessor(test_config)

def test_audio_processing(preprocessor, test_config, test_audio_file):
    """Test the complete audio preprocessing pipeline"""
    # Process all files
    processed_files = preprocessor.process_directory()
    
    # Check that files were processed
    assert len(processed_files) > 0
    
    # Test loading and validating processed files
    for proc_path in processed_files:
        # Load processed data
        data = torch.load(proc_path)
        
        # Check data structure
        assert 'spectrogram' in data
        assert 'original_path' in data
        assert 'sample_rate' in data
        
        # Check spectrogram properties
        spec = data['spectrogram']
        assert isinstance(spec, torch.Tensor)
        assert spec.dim() == 3  # [channels, mels, time]
        assert spec.shape[1] == test_config.n_mels
        
        # Test reconstruction
        audio = preprocessor.format.spectrogram_to_audio(spec)
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 2  # [channels, samples]
        assert not torch.isnan(audio).any()

def test_error_handling(preprocessor, test_audio_file):
    """Test handling of invalid audio files"""
    # Create an invalid audio file
    invalid_path = Path(preprocessor.input_dir) / "invalid.wav"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(invalid_path, 'w') as f:
        f.write("Not an audio file")
    
    # Should skip invalid file without crashing
    try:
        preprocessor.process_directory()
    finally:
        # Cleanup
        invalid_path.unlink()

def test_output_structure(preprocessor, test_audio_file):
    """Test output directory structure matches input"""
    # Process files
    processed_files = preprocessor.process_directory()
    
    # Check output structure
    for proc_path in processed_files:
        # Get relative path from output dir
        rel_path = proc_path.relative_to(preprocessor.output_dir)
        
        # Check that original audio file exists
        orig_path = preprocessor.input_dir / rel_path.with_suffix('.wav')
        assert orig_path.exists() or orig_path.with_suffix('.mp3').exists()