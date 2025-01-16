"""
This file implements the process_dataset_latents script for the YT_DIFF model.

This script pre-encodes audio into VAE latent space for a dataset, and updates the audio file metadata with the latent similarity scores.

Key components and workflow:

1. Configuration:
   - Loads dataset processing configuration from JSON
   - Configures VAE encoding parameters:
     - Model selection and path
     - Batch sizes and device placement
     - Augmentation settings (time/pitch/stereo)
     - Quantization options

2. Model Setup:
   - Initializes DualDiffusion pipeline with specified VAE
   - Validates VAE has been trained
   - Moves model to appropriate device
   - Configures precision (bfloat16)

3. Audio Processing:
   - Loads audio files from dataset
   - Applies configured augmentations:
     - Multiple time offsets
     - Pitch shifting
     - Stereo mirroring
   - Converts to spectrogram format

4. Latent Generation:
   - Encodes spectrograms through VAE encoder
   - Processes in configured batch sizes
   - Optionally quantizes latent values
   - Caches latents in output files

5. Metadata Management:
   - Updates audio file metadata with:
     - Latent presence flags
     - Augmentation settings used
     - Processing status
   - Saves modified latent files

6. Distributed Processing:
   - Handles multi-GPU processing via PartialState
   - Coordinates across processes
   - Manages device placement and memory

The script provides automated latent space encoding and augmentation for audio datasets used in training.

"""