import pytest
import torch
from src.models.vae import AudioVAE, ConvBlock
from src.configs.model_config import VAEOnlyConfig

@pytest.fixture
def device():
    """Get optimal device for testing"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

@pytest.fixture
def vae_config():
    """Get VAE configuration"""
    return VAEOnlyConfig()

@pytest.fixture
def vae_model(device, vae_config):
    """Initialize VAE model"""
    model = AudioVAE(vae_config)
    return model.to(device)

@pytest.fixture
def sample_batch(device, vae_config):
    """Create a sample batch of spectrograms"""
    batch_size = 4
    n_mels = vae_config.audio.n_mels
    return torch.randn(
        batch_size, 
        vae_config.vae.in_channels, 
        n_mels, 
        n_mels
    ).to(device)

def test_conv_block():
    """Test ConvBlock functionality"""
    in_channels, out_channels = 1, 32
    block = ConvBlock(in_channels, out_channels)
    x = torch.randn(1, in_channels, 32, 32)
    out = block(x)
    assert out.shape == (1, out_channels, 32, 32)
    assert isinstance(block.conv[1], torch.nn.BatchNorm2d)
    assert isinstance(block.conv[2], torch.nn.LeakyReLU)

def test_vae_initialization(vae_model, vae_config):
    """Test VAE initialization and architecture"""
    # Test basic initialization
    assert isinstance(vae_model, AudioVAE)
    assert vae_model.latent_dim == vae_config.vae.latent_dim
    assert vae_model.in_channels == vae_config.vae.in_channels
    
    # Test encoder architecture
    assert isinstance(vae_model.encoder, torch.nn.Sequential)
    assert isinstance(vae_model.decoder, torch.nn.Sequential)
    assert isinstance(vae_model.fc_mu, torch.nn.Linear)
    assert isinstance(vae_model.fc_var, torch.nn.Linear)

def test_vae_encode(vae_model, sample_batch):
    """Test encoding functionality"""
    mu, log_var = vae_model.encode(sample_batch)
    
    # Check shapes
    assert mu.shape[0] == sample_batch.shape[0]  # Batch size matches
    assert mu.shape[1] == vae_model.latent_dim   # Latent dimension correct
    assert mu.shape == log_var.shape             # Mu and log_var have same shape
    
    # Check values are reasonable
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()
    print(f"\nEncoding Test:")
    print(f"Input shape: {sample_batch.shape}")
    print(f"Latent mean range: {mu.min():.3f} to {mu.max():.3f}")
    print(f"Latent variance range: {log_var.exp().min():.3f} to {log_var.exp().max():.3f}")

def test_vae_decode(vae_model, sample_batch):
    """Test decoding functionality"""
    # First encode to get latent vectors
    mu, log_var = vae_model.encode(sample_batch)
    z = vae_model.reparameterize(mu, log_var)
    
    # Test decode
    reconstruction = vae_model.decode(z)
    assert reconstruction.shape == sample_batch.shape
    assert not torch.isnan(reconstruction).any()

def test_vae_forward(vae_model, sample_batch):
    """Test full forward pass"""
    reconstruction, original, mu, log_var = vae_model.forward(sample_batch)
    
    # Check shapes
    assert reconstruction.shape == original.shape
    assert mu.shape[1] == vae_model.latent_dim
    assert log_var.shape[1] == vae_model.latent_dim
    
    # Check values
    assert not torch.isnan(reconstruction).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()

def test_vae_loss(vae_model, sample_batch):
    """Test loss computation"""
    reconstruction, original, mu, log_var = vae_model.forward(sample_batch)
    loss, recons_loss, kld_loss = vae_model.loss_function(
        reconstruction, original, mu, log_var
    )
    
    # Check loss values are reasonable
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert recons_loss.item() > 0
    assert kld_loss.item() > 0
    
    # Test with custom KLD weight
    custom_weight = 0.5
    loss_custom, _, _ = vae_model.loss_function(
        reconstruction, original, mu, log_var, 
        kld_weight=custom_weight
    )
    assert loss_custom.item() != loss.item()  # Should be different with different weight

def test_parameter_count(vae_model):
    """Test parameter counting functionality"""
    param_count = vae_model.count_parameters()
    assert param_count > 0
    print(f"Model has {param_count:,} parameters")

def test_device_handling(vae_model, device):
    """Test device placement"""
    # Convert both devices to base type for comparison
    model_device = vae_model.get_device()
    expected_device = torch.device(device.type)
    assert model_device == expected_device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])