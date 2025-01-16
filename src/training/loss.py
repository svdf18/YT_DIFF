"""
This file implements loss functions for training the YT_DIFF model.

Key components:

1. Multiscale Spectral Loss:
   - Computes spectral loss across multiple window sizes
   - Uses STFT to analyze frequency content
   - Handles stereo audio with separation weighting
   
2. Loss Configuration:
   - MultiscaleSpectralLoss1DConfig defines parameters
   - Controls window sizes, overlap, weighting etc.
   - Configures spectral analysis settings

3. Loss Calculation:
   - Compares generated and target spectrograms
   - Applies appropriate weighting and scaling
   - Handles mono and stereo cases differently

4. Helper Functions:
   - STFT and mel-scale conversions
   - Window function generation
   - Loss aggregation across scales
"""