import torch
import os
from training.dataset import AudioDataset

def test_dataset():
    # Create a test config (empty dict for now)
    config = {}
    
    # Point to your audio directory
    data_dir = "audio_files"
    
    # Create dataset
    dataset = AudioDataset(data_dir, config)
    
    # Print device information
    print(f"Device availability:")
    print(f"- CUDA: {torch.cuda.is_available()}")
    print(f"- MPS: {torch.backends.mps.is_available()}")
    print(f"- CPU: Always available")

    # Get the device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Test if we can load all items
    if len(dataset) > 0:
        print(f"\nFound {len(dataset)} audio files:")
        for i in range(len(dataset)):
            item = dataset[i]
            print(f"\nAudio {i+1}:")
            print(f"- Shape: {item.shape}")
            print(f"- Duration (samples): {item.shape[1]}")
            print(f"- Channels: {item.shape[0]}")
            print(f"- Type: {item.dtype}")
            
            # Move to device and back as a test
            item = item.to(device)
            item = item.cpu()  # Move back to CPU for memory efficiency
    else:
        print("\nNo audio files found in directory")

if __name__ == "__main__":
    test_dataset()