# src/utils/device.py
"""
This file contains the device initialization and configuration management for the YT_DIFF model framework.
"""

import torch
import os
import logging

def get_device():
    """
    Get the appropriate device for computation with fallback handling.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if torch.backends.mps.is_available():
        # For certain operations that aren't supported on MPS,
        # we'll need to fall back to CPU
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        logger.info("MPS device available. CPU fallback enabled for unsupported operations.")
        return torch.device("cpu")  # Using CPU by default for better compatibility
    
    logger.info("Using CPU device.")
    return torch.device("cpu")

def init_training():
    """Initialize training settings."""
    # Set precision - MPS works best with float32
    torch.set_default_dtype(torch.float32)
    
    if torch.backends.mps.is_available():
        # Enable fallback warnings to help debug MPS issues
        torch.backends.mps.enable_fallback_warnings = True

# Create a device object that can be imported
device = get_device()