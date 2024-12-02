import pytest
import torch
from pathlib import Path
from src.training.training_pipeline import train
from src.configs.base_config import TrainingConfig
from src.configs.model_config import VAEConfig

@pytest.fixture
def train_config():
    return VAEConfig(
        data_dir="src/tests/test_data/processed",
        batch_size=2,
        num_epochs=2,
        save_interval=1  # Save checkpoint every epoch for testing
    )

def test_training_loop(train_config):
    """Test if training runs without errors"""
    # Ensure checkpoint directory exists
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Verify test data exists
    test_data_dir = Path(train_config.data_dir)
    assert test_data_dir.exists(), f"Test data directory {test_data_dir} does not exist"
    assert any(test_data_dir.iterdir()), f"Test data directory {test_data_dir} is empty"
    
    try:
        train(train_config)
        
        # Check if checkpoint was saved
        checkpoint_files = list(Path("checkpoints").glob("model_epoch_*.pt"))
        assert len(checkpoint_files) > 0
        
        # Load and verify checkpoint structure
        checkpoint = torch.load(checkpoint_files[0])
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'loss' in checkpoint
        
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")