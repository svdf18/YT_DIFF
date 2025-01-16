"""
This file implements the process_dataset_embeddings script for the YT_DIFF model.

This script pre-encodes audio and text embeddings for a dataset, and updates the audio file metadata with the embedding similarity scores.

Key components and workflow:

1. Configuration:
   - Loads dataset processing configuration from JSON
   - Configures CLAP embedding parameters:
     - Audio/text encoder models
     - Label/tag definitions
     - Batch sizes and compilation options
     - Force re-encoding flags

2. CLAP Model Setup:
   - Initializes CLAP model with specified audio/text encoders
   - Enables embedding fusion if configured
   - Loads pretrained weights
   - Moves model to appropriate device

3. Audio Embedding Generation:
   - Loads audio files from dataset
   - Processes audio through CLAP audio encoder
   - Generates embeddings in batches
   - Caches embeddings in latent files
   - Updates metadata with embedding info

4. Text Embedding Generation:
   - Processes configured labels and tags
   - Generates text embeddings via CLAP text encoder
   - Caches text embeddings
   - Computes similarity scores between audio/text

5. Metadata Management:
   - Updates audio file metadata with:
     - Embedding presence flags
     - Similarity scores
     - Processing status
   - Saves modified latent files

6. Distributed Processing:
   - Handles multi-GPU processing via PartialState
   - Coordinates across processes
   - Manages device placement

The script provides automated embedding generation and metadata updates for audio datasets used in training.

"""
