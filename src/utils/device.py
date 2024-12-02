# src/utils/device.py
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Create a device object that can be imported
device = get_device()

def init_training():
    # Set precision - MPS works best with float32 for now
    torch.set_default_dtype(torch.float32)