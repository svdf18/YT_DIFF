import torch
import os
from pathlib import Path

def inspect_pt_file(file_path):
    # Load the .pt file
    data = torch.load(file_path)
    
    # Extract audio tensor and metadata
    audio = data['audio']
    metadata = data['metadata']
    
    print(f"\nFile: {file_path}")
    print(f"Audio Shape: {audio.shape}")  # Will show [2, samples] for stereo
    print(f"Sample Rate: 32000 Hz")  # We know this from processing
    print(f"Duration: {audio.shape[1]/32000:.2f} seconds")
    print(f"Category: {metadata['category']}")
    print(f"Labels: {metadata['labels']}")
    print(f"Spectrogram Shape: {audio.shape}")  # From SpectrogramFormat processor
    print(f"Format: {metadata['format']}")  # Should be 'spectrogram'
    print("-" * 50)

def inspect_processed_directory(processed_dir):
    """Inspect processed files in flat category structure"""
    print("\nInspecting processed audio files:")
    print("=" * 50)
    
    # Get all categories (directories)
    categories = sorted([d for d in os.listdir(processed_dir) 
                       if os.path.isdir(os.path.join(processed_dir, d))])
    
    for category in categories:
        category_path = Path(processed_dir) / category
        print(f"\n=== Category: {category} ===")
        
        # Get all .pt files in this category
        pt_files = sorted(list(category_path.glob("*.pt")))
        
        if not pt_files:
            print("No processed files found")
            continue
            
        # Print file count
        print(f"Found {len(pt_files)} files")
        
        # Inspect each file
        for pt_file in pt_files:
            inspect_pt_file(pt_file)

if __name__ == "__main__":
    processed_dir = "dataset/processed"
    inspect_processed_directory(processed_dir)
