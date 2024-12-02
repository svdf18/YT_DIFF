import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

class SpectrogramFormat:
    """
    Handles conversion between audio waveforms and mel spectrograms.
    
    This class provides methods to:
    1. Convert audio to mel spectrograms (for model input)
    2. Convert mel spectrograms back to audio (for generation/reconstruction)
    3. Handle audio normalization and preprocessing
    """
    
    def __init__(self, config):
        """
        Initialize the spectrogram converter with given configuration.
        
        Args:
            config (AudioConfig): Configuration containing:
                - sample_rate: Audio sample rate (default: 44100 Hz)
                - n_fft: Size of FFT window (default: 2048 samples)
                - n_mels: Number of mel bands (default: 80 bands)
                - hop_length: Number of samples between FFT windows (default: 512 samples)
        """
        # Extract configuration parameters with defaults
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.hop_length = config.hop_length
        
        # Initialize mel spectrogram converter
        # This transforms audio waveforms to mel spectrograms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,    # Audio sample rate
            n_fft=self.n_fft,               # Length of FFT window
            n_mels=self.n_mels,             # Number of mel filterbanks
            hop_length=self.hop_length,      # Number of samples between windows
            normalized=True,                 # Normalize mel filterbanks
            power=2.0                        # Power spectrogram (2.0 for power, 1.0 for magnitude)
        )
        
        # Initialize inverse mel scale converter
        # This helps convert mel spectrograms back to linear spectrograms
        self.inverse_mel_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,     # Number of FFT bins
            n_mels=self.n_mels,             # Number of mel filterbanks
            sample_rate=self.sample_rate,    # Audio sample rate
        )
        
        # Initialize Griffin-Lim algorithm for phase reconstruction
        # This helps convert spectrograms back to audio by estimating phase
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,                       # Should match mel_transform power
            n_iter=32                        # Number of iterations for phase estimation
        )
        
    def audio_to_spectrogram(self, audio):
        """
        Convert audio waveform to mel spectrogram.
        
        Process:
        1. Convert stereo to mono if necessary
        2. Transform to mel spectrogram
        3. Convert to log scale for better neural network processing
        
        Args:
            audio (Tensor): Audio waveform [channels, samples]
        
        Returns:
            Tensor: Log mel spectrogram [channels, n_mels, time]
        """
        # Convert stereo to mono by averaging channels
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Ensure audio is normalized
        audio = self.normalize_audio(audio)
        
        # Convert to mel spectrogram
        # Shape: [channels, n_mels, time]
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale (add small constant to avoid log(0))
        # This better matches human perception and helps neural network training
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec

    def spectrogram_to_audio(self, spec):
        """
        Convert mel spectrogram back to audio using Griffin-Lim algorithm.
        
        Process:
        1. Convert from log scale back to linear
        2. Convert mel spec to linear spec
        3. Reconstruct phase information
        4. Convert to audio waveform
        
        Args:
            spec (Tensor): Log mel spectrogram [channels, n_mels, time]
        
        Returns:
            Tensor: Audio waveform [channels, samples]
        """
        # Convert from log scale back to linear
        spec = torch.exp(spec)
        
        # Convert mel spec to linear spec
        # This recovers the linear frequency scale spectrogram
        linear_spec = self.inverse_mel_transform(spec)
        
        # Reconstruct phase information and convert to audio
        # Griffin-Lim iteratively estimates the phase information
        audio = self.griffin_lim(linear_spec)
        
        # Ensure output audio is normalized
        audio = self.normalize_audio(audio)
        
        return audio

    def normalize_audio(self, audio):
        """
        Normalize audio to the range [-1, 1].
        
        Args:
            audio (Tensor): Input audio waveform
            
        Returns:
            Tensor: Normalized audio waveform
        """
        # Normalize using the infinity norm (maximum absolute value)
        return F.normalize(audio, p=float('inf'), dim=1)