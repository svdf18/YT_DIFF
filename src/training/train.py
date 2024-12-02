import argparse
from src.training.training_pipeline import train
from src.configs.base_config import TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Create config with CLI arguments
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    train(config)

if __name__ == "__main__":
    main()