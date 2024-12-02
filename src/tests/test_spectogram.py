import torch
import torchaudio
import unittest
import os
from modules.spectogram import SpectrogramFormat

class TestSpectrogramFormat(unittest.TestCase):
    """
    Test suite for SpectrogramFormat class.
    Tests audio-to-spectrogram conversion, reconstruction, and normalization.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        Creates a SpectrogramFormat instance and loads test audio.
        """
        # Test configuration
        self.config = {
            'sample_rate': 44100,
            'n_fft': 2048,
            'n_mels': 80,
            'hop_length': 512
        }
        
        # Initialize SpectrogramFormat
        self.spec_format = SpectrogramFormat(self.config)
        
        # Load test audio
        audio_path = "audio_files"  
        self.test_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                          if f.endswith(('.wav', '.mp3'))]
        
        if not self.test_files:
            raise ValueError("No test audio files found in audio_files directory")
            
        # Load first audio file for testing
        self.audio, self.sr = torchaudio.load(self.test_files[0])
        
    def test_audio_normalization(self):
        """Test if audio normalization works correctly."""
        normalized = self.spec_format.normalize_audio(self.audio)
        
        # Check if values are in [-1, 1] range
        self.assertTrue(torch.all(normalized >= -1))
        self.assertTrue(torch.all(normalized <= 1))
        
        # Check if at least one value is close to 1 or -1 (proper normalization)
        self.assertTrue(torch.max(torch.abs(normalized)).item() > 0.9)
        
    def test_spectrogram_conversion(self):
        """Test audio to spectrogram conversion."""
        # Convert to spectrogram
        spec = self.spec_format.audio_to_spectrogram(self.audio)
        
        # Check output shape
        expected_time_steps = (self.audio.shape[1] // self.config['hop_length']) + 1
        self.assertEqual(spec.shape[1], self.config['n_mels'])  # Frequency bins
        self.assertGreater(spec.shape[2], 0)  # Time steps
        
        # Check if values are in reasonable range for log spectrograms
        self.assertTrue(torch.all(torch.isfinite(spec)))  # No NaN or inf values
        
    def test_audio_reconstruction(self):
        """Test spectrogram to audio reconstruction."""
        # Convert to spectrogram and back
        spec = self.spec_format.audio_to_spectrogram(self.audio)
        reconstructed = self.spec_format.spectrogram_to_audio(spec)
        
        # Check if reconstructed audio has similar length
        length_diff = abs(self.audio.shape[1] - reconstructed.shape[1])
        max_allowed_diff = self.config['hop_length'] * 2  # Allow some difference due to reconstruction
        self.assertLess(length_diff, max_allowed_diff)
        
        # Check if reconstructed audio is normalized
        self.assertTrue(torch.all(reconstructed >= -1))
        self.assertTrue(torch.all(reconstructed <= 1))
        
    def test_stereo_handling(self):
        """Test handling of stereo audio input."""
        # Create fake stereo audio if input is mono
        if self.audio.shape[0] == 1:
            stereo_audio = torch.cat([self.audio, self.audio], dim=0)
        else:
            stereo_audio = self.audio
            
        # Convert stereo to spectrogram
        spec = self.spec_format.audio_to_spectrogram(stereo_audio)
        
        # Check if output is mono (averaged channels)
        self.assertEqual(spec.shape[0], 1)
        
    def test_batch_processing(self):
        """Test processing of multiple audio files."""
        # Load all test files
        audio_batch = []
        for file in self.test_files[:3]:  # Test with first 3 files
            audio, sr = torchaudio.load(file)
            audio_batch.append(audio)
            
        # Process each file
        specs = []
        for audio in audio_batch:
            spec = self.spec_format.audio_to_spectrogram(audio)
            specs.append(spec)
            
        # Check if all spectrograms have the same number of mel bins
        mel_bins = [spec.shape[1] for spec in specs]
        self.assertEqual(len(set(mel_bins)), 1)
        self.assertEqual(mel_bins[0], self.config['n_mels'])

if __name__ == '__main__':
    unittest.main()