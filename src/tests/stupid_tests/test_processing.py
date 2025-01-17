from simple_processor import SimpleAudioProcessor
import os

if __name__ == "__main__":
    # Setup paths
    config_path = "config/dataset/dataset.json"
    input_dir = "dataset/raw_audio"
    output_dir = "dataset/processed"
    
    # Create processor and run
    processor = SimpleAudioProcessor(config_path)
    processor.process_dataset(input_dir, output_dir)
