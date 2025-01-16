"""
This file implements the dataset processor class for the YT_DIFF model.

The dataset processor class is responsible for loading, processing, and splitting the dataset into training and validation sets.

Key components and workflow:

1. Dataset Configuration:
   - Defines configuration parameters for dataset processing via DatasetProcessorConfig
   - Specifies supported audio formats, sample rates, channels
   - Controls data filtering criteria (min/max lengths, quality thresholds)
   - Configures data augmentation options

2. Audio Processing:
   - Loads audio files in various formats (wav, mp3, etc)
   - Validates audio quality metrics (sample rate, bitrate)
   - Resamples and normalizes audio to target format
   - Handles mono/stereo channel conversion

3. Data Preprocessing:
   - Pre-encodes audio into VAE latent space (optional)
   - Generates data augmentations:
     - Time offsets
     - Pitch shifts 
     - Stereo mirroring
   - Applies quantization if enabled

4. Embedding Generation:
   - Extracts CLAP embeddings for audio and text
   - Supports different CLAP model variants
   - Enables embedding fusion
   - Caches embeddings for reuse
   - Handles batched processing

5. Dataset Organization:
   - Splits data into train/validation sets
   - Enforces minimum samples per class
   - Manages dataset metadata and caching
   - Tracks processing statistics

6. Utility Functions:
   - Audio format conversion
   - Quality validation
   - Progress tracking
   - Metadata management
   - Cache handling

The processor provides a unified interface for preparing audio datasets for training the YT_DIFF model, with configurable preprocessing, augmentation, and embedding generation.

"""