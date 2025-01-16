"""
This file implements Power Function EMA (Exponential Moving Average) for model training.

Key components:

1. EMA Manager:
   - Maintains multiple EMA models with different decay rates (betas)
   - Handles model parameter updates during training
   - Manages saving/loading of EMA checkpoints

2. EMA Updates:
   - Applies exponential moving average to model parameters
   - Uses warmup period to gradually increase EMA effect
   - Supports multiple beta values for different averaging timescales
"""