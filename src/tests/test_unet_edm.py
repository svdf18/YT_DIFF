import pytest
import torch
from src.models.unet_edm import TimeEmbedding, ResnetBlock, AttentionBlock, UNetEDM
from src.utils.device import device

def test_time_embedding():
    """Test the TimeEmbedding module"""
    batch_size = 4
    dim = 256
    embedding = TimeEmbedding(dim).to(device)
    
    # Test with random sigma values
    sigma = torch.rand(batch_size).to(device)
    output = embedding(sigma)
    
    assert output.shape == (batch_size, dim)
    assert not torch.isnan(output).any()

def test_resnet_block():
    """Test the ResnetBlock module"""
    batch_size = 4
    channels = 64
    time_dim = 256
    spatial_size = 32
    
    block = ResnetBlock(channels, channels, time_dim).to(device)
    
    x = torch.randn(batch_size, channels, spatial_size, spatial_size).to(device)
    t = torch.randn(batch_size, time_dim).to(device)
    
    output = block(x, t)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

def test_attention_block():
    """Test the AttentionBlock module"""
    batch_size = 4
    channels = 64
    spatial_size = 32
    
    block = AttentionBlock(channels).to(device)
    x = torch.randn(batch_size, channels, spatial_size, spatial_size).to(device)
    
    output = block(x)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

def test_unet_edm():
    """Test the complete UNetEDM model"""
    batch_size = 4
    in_channels = 1
    spatial_size = 32
    
    config = {
        'channels': [64, 128, 256, 512],
        'time_dim': 256,
        'in_channels': in_channels
    }
    
    model = UNetEDM(config).to(device)
    
    x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
    sigma = torch.rand(batch_size).to(device)
    
    output = model(x, sigma)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

def test_unet_edm_gradient_flow():
    """Test gradient flow through the UNetEDM model"""
    batch_size = 4
    in_channels = 1
    spatial_size = 32
    
    config = {
        'channels': [64, 128, 256, 512],
        'time_dim': 256,
        'in_channels': in_channels
    }
    
    model = UNetEDM(config).to(device)
    
    x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
    sigma = torch.rand(batch_size).to(device)
    
    # Forward pass
    output = model(x, sigma)
    
    # Compute loss and backward pass
    loss = output.mean()
    loss.backward()
    
    # Check if gradients exist and are not nan
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

def test_different_input_sizes():
    """Test UNetEDM with different input sizes"""
    config = {
        'channels': [64, 128, 256, 512],
        'time_dim': 256,
        'in_channels': 1
    }
    
    model = UNetEDM(config).to(device)
    
    # Test different spatial sizes that are powers of 2
    for size in [16, 32, 64]:
        x = torch.randn(2, 1, size, size).to(device)
        sigma = torch.rand(2).to(device)
        
        output = model(x, sigma)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

def test_model_memory():
    """Test memory usage of the model"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    config = {
        'channels': [64, 128, 256, 512],
        'time_dim': 256,
        'in_channels': 1
    }
    
    model = UNetEDM(config).to(device)
    
    mem_after = process.memory_info().rss
    mem_diff_mb = (mem_after - mem_before) / (1024 * 1024)
    
    # Model should not use more than 1GB of memory
    assert mem_diff_mb < 1024, f"Model uses {mem_diff_mb:.2f}MB of memory"

if __name__ == "__main__":
    pytest.main([__file__])