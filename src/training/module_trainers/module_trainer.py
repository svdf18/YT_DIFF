"""
This file implements the module trainer class for the YT_DIFF model.

The training process for each module follows these key steps:

1. Batch Initialization (init_batch):
   - Prepares module state for training
   - Resets gradients and internal buffers
   - Sets training mode flags
   - Handles validation mode setup if needed

2. Training Step (train_batch):
   - Processes input batch data
   - Computes forward pass and loss
   - Accumulates gradients if using gradient accumulation
   - Returns dict of loss metrics
   - Handles device placement and mixed precision

3. Batch Finalization (finish_batch):
   - Updates model parameters with accumulated gradients
   - Applies gradient clipping if configured
   - Updates learning rate schedules
   - Returns final batch metrics
   - Resets internal state for next batch

4. Validation:
   - Disables training-specific operations
   - Computes metrics without gradient updates
   - Uses same pipeline as training for consistency

The ModuleTrainer base class defines this interface that concrete
module trainers must implement to fit into the training framework.

"""