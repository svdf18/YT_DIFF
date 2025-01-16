"""
This file implements the trainer class for the dual diffusion model.

The training process follows these key steps:

1. Model Initialization:
   - Loads model architecture and weights
   - Sets up optimizers and learning rate schedules
   - Initializes EMA (Exponential Moving Average) for model weights

2. Data Preparation:
   - Creates data loaders and preprocessing pipelines
   - Handles batching and device placement
   - Manages caching of processed data

3. Training Loop:
   - Iterates through epochs and batches
   - Computes loss and gradients
   - Updates model parameters
   - Logs metrics and checkpoints

4. Validation:
   - Periodically evaluates model on validation set
   - Tracks key metrics like loss and FID score
   - Saves best performing checkpoints

5. Logging & Monitoring:
   - Records training progress and metrics
   - Generates sample outputs for inspection
   - Handles early stopping if needed

6. Cleanup & Finalization:
   - Saves final model weights and config
   - Exports training logs and artifacts
   - Releases compute resources

"""