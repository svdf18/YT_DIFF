import argparse
from src.training.training_pipeline import train
from src.configs.model_config import create_config
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_type', type=str, choices=['vae', 'edm2', 'hybrid'], required=True)
    args = parser.parse_args()
    
    # Create config with CLI arguments
    config = create_config(args.model_type)
    
    # Set data directories - Correct the path construction
    config.training.train_dir = Path(args.data_dir) / "training" / "processed"
    config.training.val_dir = Path(args.data_dir) / "validation" / "processed"
    
    # Update other training parameters
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    
    # Verify directories exist
    if not config.training.train_dir.exists():
        raise ValueError(f"Training directory not found: {config.training.train_dir}")
    if not config.training.val_dir.exists():
        raise ValueError(f"Validation directory not found: {config.training.val_dir}")
    
    train(config)

if __name__ == "__main__":
    main()