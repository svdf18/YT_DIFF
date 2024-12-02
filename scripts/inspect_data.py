import torch
from pathlib import Path
import argparse

def inspect_processed_file(file_path):
    """Inspect a processed spectrogram file"""
    try:
        data = torch.load(file_path)
        
        if isinstance(data, dict):
            spec = data['spectrogram']
            return {
                'file_path': file_path,
                'shape': spec.shape,
                'min': spec.min().item(),
                'max': spec.max().item(),
                'mean': spec.mean().item(),
                'original_path': data['original_path'],
                'sample_rate': data['sample_rate']
            }
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_files', type=int, default=5, help='Number of files to inspect')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all processed files
    processed_files = []
    for ext in ['*.pt', '*.npy']:
        processed_files.extend(data_dir.glob(ext))
    
    print(f"Found {len(processed_files)} processed files in {data_dir}")
    
    # Inspect files
    inspected_data = []
    for file_path in list(processed_files)[:args.num_files]:
        result = inspect_processed_file(file_path)
        if result:
            inspected_data.append(result)
    
    # Print summary
    for data in inspected_data:
        print(f"\nFile: {data['file_path']}")
        print(f"- Original Path: {data['original_path']}")
        print(f"- Sample Rate: {data['sample_rate']}")
        print(f"- Shape: {data['shape']}")
        print(f"- Min: {data['min']:.3f}, Max: {data['max']:.3f}, Mean: {data['mean']:.3f}")

if __name__ == "__main__":
    main()