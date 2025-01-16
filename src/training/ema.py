"""
This file implements Exponential Moving Average (EMA) functionality for model training.

Key components:

1. EMA Manager:
   - Maintains multiple EMA models with different decay rates (betas)
   - Handles model parameter updates during training
   - Manages saving/loading of EMA checkpoints

2. EMA Updates:
   - Applies exponential moving average to model parameters
   - Uses warmup period to gradually increase EMA effect
   - Supports multiple beta values for different averaging timescales

3. Checkpoint Management:
   - Saves/loads EMA models to/from disk
   - Handles device placement and memory management
   - Maintains sorted list of EMA checkpoints

4. Helper Functions:
   - get_ema_list: Lists available EMA checkpoints
   - Parameter copying and state management
   - Device placement utilities
"""
