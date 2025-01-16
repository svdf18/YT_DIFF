"""
This file implements the dual diffusion pipeline class for the YT_DIFF model.

The pipeline class provides a unified interface for loading, training, and sampling from the YT_DIFF model.
It supports loading pre-trained models, compiling modules for performance, and performing inference with various sampling methods.
Key components and workflow:

1. Pipeline Configuration:
   - Defines a SampleParams dataclass for configuring generation parameters
   - Handles model loading, compilation and device placement
   - Manages training and inference modes

2. Model Components:
   - Integrates VAE for encoding/decoding audio
   - Uses UNet for denoising diffusion process
   - Supports text conditioning via embeddings

3. Sampling Process:
   - Implements various sampling schedules (EDM, DDPM etc)
   - Handles batched generation with configurable steps
   - Supports classifier-free guidance for controlled generation
   - Enables img2img and inpainting workflows

4. Audio Processing:
   - Handles audio loading and preprocessing
   - Manages seamless loop generation
   - Supports variable length generation
   - Processes input conditioning signals

5. Training Interface:
   - Provides unified training loop for all modules
   - Handles gradient accumulation and mixed precision
   - Manages EMA parameter averaging
   - Tracks training metrics and checkpoints

6. Inference Features:
   - Batched parallel sampling
   - Configurable noise schedules
   - Support for conditional generation
   - Various sampling methods (DDPM, DDIM, etc)

7. Utility Functions:
   - Audio format conversion
   - Model state management
   - Device placement optimization
   - Progress tracking and logging


"""